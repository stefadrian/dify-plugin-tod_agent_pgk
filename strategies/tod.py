import json
import time
from typing import Any, Generator, Optional

from dify_plugin.entities.agent import AgentInvokeMessage
from dify_plugin.entities.model.llm import LLMModelConfig, LLMUsage
from dify_plugin.entities.model.message import SystemPromptMessage, UserPromptMessage
from dify_plugin.interfaces.agent import AgentStrategy
from dify_plugin.entities.tool import ToolInvokeMessage, LogMetadata

from .models import DialogueState, DialogueField, TODParams
from .utils import try_parse_json, increase_usage
from .prompts import tod_system_prompt, tod_extract_prompt

class TaskOrientedDialogueStrategy(AgentStrategy):
    def __init__(self, session):
        super().__init__(session)
        self.dialogue_state = None
        self.collected_data = {}
        self.current_model_config = None
        self.llm_usage: dict[str, Optional[LLMUsage]] = {"usage": None}

    def _get_conversation_history(self) -> str:
        history = []
        for field in self.dialogue_state.fields[
            : self.dialogue_state.current_field_index
        ]:
            if field.value:
                history.append(f"Q: {field.question}\nA: {field.value}")
        return "\n".join(history)

    def _init_dialogue_state(self, information_schema: str) -> DialogueState:
        try:
            schema = json.loads(information_schema)
            fields = []
            for field in schema.get("fields", []):
                fields.append(
                    DialogueField(
                        name=field["name"],
                        question=field["question"],
                        required=field.get("required", True),
                    )
                )
            return DialogueState(fields=fields)
        except Exception:
            raise ValueError("Failed to initialize dialogue state")

    def _load_dialogue_state(self, storage_key: str) -> Optional[DialogueState]:
        try:
            if stored_data := self.session.storage.get(storage_key):
                state_dict = json.loads(stored_data.decode())
                dialogue_state = DialogueState(**state_dict)
                self.collected_data = {
                    field.name: field.value
                    for field in dialogue_state.fields
                    if field.value is not None
                }
                return dialogue_state
        except Exception:
            return None

    def _save_dialogue_state(self, storage_key: str):
        try:
            if self.dialogue_state:
                state_dict = self.dialogue_state.model_dump()
                self.session.storage.set(storage_key, json.dumps(state_dict).encode())
        except Exception:
            pass

    def _extract_answers(self, user_input: str, parent_log=None) -> Generator[AgentInvokeMessage, None, None]:
        extract_prompt = tod_extract_prompt(self.dialogue_state.fields, user_input)

        extract_started_at = time.perf_counter()
        extract_log = self.create_log_message(
            label=f"Answer Extraction",
            data={
                "user_input": user_input,
                "extract_prompt": extract_prompt,
                "type": "extraction"
            },
            metadata={
                LogMetadata.STARTED_AT: extract_started_at,
            },
            parent=parent_log,
            status=ToolInvokeMessage.LogMessage.LogStatus.START
        )
        yield extract_log

        model_config = LLMModelConfig(**self.current_model_config.model_dump(mode="json"))
        response = self.session.model.llm.invoke(
            model_config=model_config,
            prompt_messages=[
                SystemPromptMessage(content=extract_prompt)
            ],
            stream=False
        )
        extracted_data = try_parse_json(response.message.content)

        increase_usage(self.llm_usage, response.usage)

        finish_log = self.finish_log_message(
            log=extract_log,
            data={
                "response": response.message.content,
                "extracted_data": extracted_data
            },
            metadata={
                LogMetadata.STARTED_AT: extract_started_at,
                LogMetadata.FINISHED_AT: time.perf_counter(),
                LogMetadata.ELAPSED_TIME: time.perf_counter() - extract_started_at,
                LogMetadata.TOTAL_TOKENS: response.usage.total_tokens if response.usage else 0,
            }
        )
        yield finish_log
        yield extracted_data

    def _invoke(self, parameters: dict[str, Any]) -> Generator[AgentInvokeMessage, None, None]:
        params = TODParams(**parameters)
        self.current_model_config = params.model
        self.llm_usage = {"usage": None}

        round_started_at = time.perf_counter()
        round_log = self.create_log_message(
            label=f"Dialogue Round",
            data={
                "query": params.query,
                "information_schema": params.information_schema,
                "conversation_history": self._get_conversation_history() if self.dialogue_state else "",
                "dialogue_state": {
                    "current_field_index": self.dialogue_state.current_field_index if self.dialogue_state else 0,
                    "total_fields": len(self.dialogue_state.fields) if self.dialogue_state else 0,
                    "completed_fields": [
                        {"name": f.name, "question": f.question, "value": f.value}
                        for f in self.dialogue_state.fields
                        if self.dialogue_state and f.value is not None
                    ]
                } if self.dialogue_state else None
            },
            metadata={
                LogMetadata.STARTED_AT: round_started_at,
            },
            status=ToolInvokeMessage.LogMessage.LogStatus.START
        )
        yield round_log

        if not self.dialogue_state:
            try:
                self.dialogue_state = self._load_dialogue_state(params.storage_key)
            except Exception:
                self.dialogue_state = None

            if not self.dialogue_state:
                self.dialogue_state = self._init_dialogue_state(params.information_schema)

        # If we already requested confirmation previously, interpret current user input as confirmation or correction
        if self.dialogue_state.confirmation_requested and not self.dialogue_state.confirmed:
            lower_q = params.query.strip().lower()
            # Simple heuristics for confirmation / denial
            positive_markers = {"yes", "y", "ok", "okay", "confirm", "confirmed", "correct"}
            negative_markers = {"no", "n", "change", "edit", "modify", "incorrect", "not correct", "wrong"}
            if any(m == lower_q or lower_q.startswith(m+" ") for m in positive_markers):
                self.dialogue_state.confirmed = True
            elif any(m == lower_q or lower_q.startswith(m+" ") for m in negative_markers):
                # Reset to allow user to supply corrections: we ask which field they want to change
                self.dialogue_state.confirmation_requested = False
                self.dialogue_state.confirmed = False
                message = "Which field would you like to modify? Please reply with field name and new value."
                yield self.create_text_message(message)
                yield self.create_json_message({
                    "execution_metadata": {
                        LogMetadata.TOTAL_PRICE: self.llm_usage["usage"].total_price if self.llm_usage["usage"] is not None else 0,
                        LogMetadata.CURRENCY: self.llm_usage["usage"].currency if self.llm_usage["usage"] is not None else "",
                        LogMetadata.TOTAL_TOKENS: self.llm_usage["usage"].total_tokens if self.llm_usage["usage"] is not None else 0,
                    }
                })
                # Finish round log early since we're not validating/extracting in this branch
                finish_log_data = {
                    "output": message,
                    "dialogue_state": {
                        "completed": False,
                        "confirmation_requested": False,
                        "confirmed": False,
                        "current_field_index": self.dialogue_state.current_field_index,
                        "total_fields": len(self.dialogue_state.fields),
                        "completed_fields": [
                            {"name": f.name, "value": f.value}
                            for f in self.dialogue_state.fields if f.value is not None
                        ]
                    }
                }
                finish_log_metadata = {
                    LogMetadata.STARTED_AT: round_started_at,
                    LogMetadata.FINISHED_AT: time.perf_counter(),
                    LogMetadata.ELAPSED_TIME: time.perf_counter() - round_started_at,
                    LogMetadata.TOTAL_TOKENS: self.llm_usage["usage"].total_tokens if self.llm_usage["usage"] else 0,
                }
                yield self.finish_log_message(
                    log=round_log,
                    data=finish_log_data,
                    metadata=finish_log_metadata
                )
                self._save_dialogue_state(params.storage_key)
                return
            else:
                # Ask for explicit yes/no
                message = "Please confirm if the collected information is correct (yes/no)."
                yield self.create_text_message(message)
                yield self.create_json_message({
                    "execution_metadata": {
                        LogMetadata.TOTAL_PRICE: self.llm_usage["usage"].total_price if self.llm_usage["usage"] is not None else 0,
                        LogMetadata.CURRENCY: self.llm_usage["usage"].currency if self.llm_usage["usage"] is not None else "",
                        LogMetadata.TOTAL_TOKENS: self.llm_usage["usage"].total_tokens if self.llm_usage["usage"] is not None else 0,
                    }
                })
                finish_log_data = {
                    "output": message,
                    "dialogue_state": {
                        "completed": False,
                        "confirmation_requested": True,
                        "confirmed": False,
                        "current_field_index": self.dialogue_state.current_field_index,
                        "total_fields": len(self.dialogue_state.fields),
                        "completed_fields": [
                            {"name": f.name, "value": f.value}
                            for f in self.dialogue_state.fields if f.value is not None
                        ]
                    }
                }
                finish_log_metadata = {
                    LogMetadata.STARTED_AT: round_started_at,
                    LogMetadata.FINISHED_AT: time.perf_counter(),
                    LogMetadata.ELAPSED_TIME: time.perf_counter() - round_started_at,
                    LogMetadata.TOTAL_TOKENS: self.llm_usage["usage"].total_tokens if self.llm_usage["usage"] else 0,
                }
                yield self.finish_log_message(
                    log=round_log,
                    data=finish_log_data,
                    metadata=finish_log_metadata
                )
                return

        # If confirmed now, finalize
        if self.dialogue_state.confirmed and not self.dialogue_state.completed and all(f.value for f in self.dialogue_state.fields):
            self.dialogue_state.completed = True
            summary = self._generate_summary()
            yield self.create_text_message("InformationCollectionCompleted:\n"+summary)
            yield self.create_json_message(self.collected_data)
            finish_log_data = {
                "output": summary,
                "collected_data": self.collected_data,
                "dialogue_state": {
                    "completed": True,
                    "confirmed": True,
                    "current_field_index": self.dialogue_state.current_field_index,
                    "total_fields": len(self.dialogue_state.fields),
                    "completed_fields": [
                        {"name": f.name, "value": f.value}
                        for f in self.dialogue_state.fields
                    ]
                }
            }
            finish_log_metadata = {
                LogMetadata.STARTED_AT: round_started_at,
                LogMetadata.FINISHED_AT: time.perf_counter(),
                LogMetadata.ELAPSED_TIME: time.perf_counter() - round_started_at,
                LogMetadata.TOTAL_TOKENS: self.llm_usage["usage"].total_tokens if self.llm_usage["usage"] else 0,
            }
            yield self.create_json_message({
                "execution_metadata": {
                    LogMetadata.TOTAL_PRICE: self.llm_usage["usage"].total_price if self.llm_usage["usage"] is not None else 0,
                    LogMetadata.CURRENCY: self.llm_usage["usage"].currency if self.llm_usage["usage"] is not None else "",
                    LogMetadata.TOTAL_TOKENS: self.llm_usage["usage"].total_tokens if self.llm_usage["usage"] is not None else 0,
                }
            })
            yield self.finish_log_message(
                log=round_log,
                data=finish_log_data,
                metadata=finish_log_metadata
            )
            self.session.storage.delete(params.storage_key)
            return

        current_field = self.dialogue_state.fields[self.dialogue_state.current_field_index]

        if params.query.strip() == "":
            message = current_field.question
            yield self.create_text_message(message)

            finish_log_data = {
                "output": message,
                "dialogue_state": {
                    "current_field": current_field.name,
                    "current_field_index": self.dialogue_state.current_field_index,
                    "total_fields": len(self.dialogue_state.fields),
                    "completed_fields": [
                        {"name": f.name, "value": f.value}
                        for f in self.dialogue_state.fields
                        if f.value is not None
                    ]
                }
            }
            finish_log_metadata = {
                LogMetadata.STARTED_AT: round_started_at,
                LogMetadata.FINISHED_AT: time.perf_counter(),
                LogMetadata.ELAPSED_TIME: time.perf_counter() - round_started_at,
                LogMetadata.TOTAL_TOKENS: 0,
            }

            yield self.finish_log_message(
                log=round_log,
                data=finish_log_data,
                metadata=finish_log_metadata
            )
            return

        # Check if this is the first interaction and user is expressing intent rather than answering a specific question
        is_first_interaction = self.dialogue_state.current_field_index == 0 and not any(f.value for f in self.dialogue_state.fields)
        
        conversation_history = self._get_conversation_history()
        context_prompt = f"""Collected information:
{conversation_history}

Current question: {current_field.question}
User answer: {params.query}"""

        system_message = SystemPromptMessage(content=tod_system_prompt())

        validation_started_at = time.perf_counter()
        validation_log = self.create_log_message(
            label=f"Answer Validation",
            data={
                "context": context_prompt,
                "system_prompt": system_message.content,
                "type": "validation",
                "is_first_interaction": is_first_interaction
            },
            metadata={
                LogMetadata.STARTED_AT: validation_started_at,
                LogMetadata.PROVIDER: params.model.provider,
            },
            parent=round_log,
            status=ToolInvokeMessage.LogMessage.LogStatus.START
        )
        yield validation_log

        model_config = LLMModelConfig(**params.model.model_dump(mode="json"))
        response = self.session.model.llm.invoke(
            model_config=model_config,
            prompt_messages=[
                system_message,
                UserPromptMessage(content=context_prompt),
            ],
            stream=False
        )

        # For first interaction, be more lenient if user is expressing intent
        if is_first_interaction:
            # Check if user is expressing intent to be contacted or similar
            intent_keywords = ["contact", "be contacted", "reach out", "get in touch", "follow up", "call me", "email me"]
            user_input_lower = params.query.lower()
            
            if any(keyword in user_input_lower for keyword in intent_keywords):
                # User is expressing intent to be contacted, start with first question
                is_valid = False  # Don't validate against current question, just start the flow
                response.message.content = f"INVALID: User is expressing intent to be contacted, starting information collection"
            else:
                is_valid = self._is_valid_answer(response.message.content)
        else:
            is_valid = self._is_valid_answer(response.message.content)
        increase_usage(self.llm_usage, response.usage)

        yield self.finish_log_message(
            log=validation_log,
            data={
                "response": response.message.content,
                "is_valid": is_valid
            },
            metadata={
                LogMetadata.STARTED_AT: validation_started_at,
                LogMetadata.FINISHED_AT: time.perf_counter(),
                LogMetadata.ELAPSED_TIME: time.perf_counter() - validation_started_at,
                LogMetadata.PROVIDER: params.model.provider,
                LogMetadata.TOTAL_TOKENS: response.usage.total_tokens if response.usage else 0,
            }
        )

        extracted_answers = {}
        if is_valid or params.query.strip() != "":
            extraction_generator = self._extract_answers(params.query, round_log)
            last_item = None
            for item in extraction_generator:
                if isinstance(item, AgentInvokeMessage):
                    yield item
                else:
                    last_item = item

            if isinstance(last_item, dict):
                extracted_answers = last_item

            answers_saved = False
            fields_to_update = {}

            for field in self.dialogue_state.fields:
                if field.name in extracted_answers and extracted_answers[field.name]:
                    fields_to_update[field.name] = extracted_answers[field.name]
                    answers_saved = True

            if answers_saved:
                for field in self.dialogue_state.fields:
                    if field.name in fields_to_update:
                        field.value = fields_to_update[field.name]
                        self.collected_data[field.name] = field.value

                next_field_index = len(self.dialogue_state.fields)
                for i, field in enumerate(self.dialogue_state.fields):
                    if field.value is None or field.value == "":
                        next_field_index = i
                        break

                self.dialogue_state.current_field_index = next_field_index
                self._save_dialogue_state(params.storage_key)

                if self.dialogue_state.current_field_index >= len(self.dialogue_state.fields):
                    # All fields collected: ask for confirmation instead of immediate completion
                    self.dialogue_state.confirmation_requested = True
                    summary = self._generate_summary()
                    message = (
                        "Please review the collected information below. If everything is correct, reply 'yes' to confirm, "
                        "or reply 'no' / indicate the field name with new value to modify.\n" + summary
                    )
                    yield self.create_text_message(message)
                    self._save_dialogue_state(params.storage_key)
                    # Do not mark completed yet; wait for user confirmation
                    yield self.create_json_message(
                        {
                            "execution_metadata": {
                                LogMetadata.TOTAL_PRICE: self.llm_usage["usage"].total_price
                                if self.llm_usage["usage"] is not None
                                else 0,
                                LogMetadata.CURRENCY: self.llm_usage["usage"].currency
                                if self.llm_usage["usage"] is not None
                                else "",
                                LogMetadata.TOTAL_TOKENS: self.llm_usage["usage"].total_tokens
                                if self.llm_usage["usage"] is not None
                                else 0,
                            }
                        }
                    )
                    finish_log_data = {
                        "output": message,
                        "collected_data": self.collected_data,
                        "dialogue_state": {
                            "completed": False,
                            "confirmation_requested": True,
                            "confirmed": False,
                            "current_field_index": self.dialogue_state.current_field_index,
                            "total_fields": len(self.dialogue_state.fields),
                            "completed_fields": [
                                {"name": f.name, "value": f.value}
                                for f in self.dialogue_state.fields
                            ]
                        }
                    }
                    finish_log_metadata = {
                        LogMetadata.STARTED_AT: round_started_at,
                        LogMetadata.FINISHED_AT: time.perf_counter(),
                        LogMetadata.ELAPSED_TIME: time.perf_counter() - round_started_at,
                        LogMetadata.TOTAL_TOKENS: self.llm_usage["usage"].total_tokens if self.llm_usage["usage"] else 0,
                    }
                    yield self.finish_log_message(
                        log=round_log,
                        data=finish_log_data,
                        metadata=finish_log_metadata
                    )
                    self._save_dialogue_state(params.storage_key)
                    return
                else:
                    # Move to next question
                    next_field = self.dialogue_state.fields[self.dialogue_state.current_field_index]
                    message = next_field.question
                    yield self.create_text_message(message)
                    
                    yield self.create_json_message(
                        {
                            "execution_metadata": {
                                LogMetadata.TOTAL_PRICE: self.llm_usage["usage"].total_price
                                if self.llm_usage["usage"] is not None
                                else 0,
                                LogMetadata.CURRENCY: self.llm_usage["usage"].currency
                                if self.llm_usage["usage"] is not None
                                else "",
                                LogMetadata.TOTAL_TOKENS: self.llm_usage["usage"].total_tokens
                                if self.llm_usage["usage"] is not None
                                else 0,
                            }
                        }
                    )

                    finish_log_data = {
                        "output": message,
                        "dialogue_state": {
                            "completed": False,
                            "current_field_index": self.dialogue_state.current_field_index,
                            "total_fields": len(self.dialogue_state.fields),
                            "completed_fields": [
                                {"name": f.name, "value": f.value}
                                for f in self.dialogue_state.fields
                                if f.value is not None
                            ],
                            "current_field": {
                                "name": next_field.name,
                                "question": next_field.question
                            }
                        },
                        "model_interactions_summary": {
                            "validation_result": is_valid,
                            "extracted_answers": extracted_answers,
                            "total_interactions": 2
                        }
                    }
                    finish_log_metadata = {
                        LogMetadata.STARTED_AT: round_started_at,
                        LogMetadata.FINISHED_AT: time.perf_counter(),
                        LogMetadata.ELAPSED_TIME: time.perf_counter() - round_started_at,
                        LogMetadata.TOTAL_TOKENS: self.llm_usage["usage"].total_tokens if self.llm_usage["usage"] else 0,
                    }

                    yield self.finish_log_message(
                        log=round_log,
                        data=finish_log_data,
                        metadata=finish_log_metadata
                    )
                    self._save_dialogue_state(params.storage_key)
                    return
            else:
                # No answers were extracted - show error or ask for clarification
                if not is_valid:
                    reason = response.message.content.split("INVALID:", 1)[1].strip() if response.message.content.startswith("INVALID:") else "Answer is not clear enough"

                    # Handle special case for first interaction with intent expression
                    if is_first_interaction and "expressing intent to be contacted" in reason:
                        message = f"I understand you want to be contacted. To help you better, {current_field.question}"
                    elif "greeting" in reason or "small talk" in reason.lower():
                        message = f"Hello! {current_field.question}"
                    elif "incomplete" in reason or "unclear" in reason.lower():
                        message = f"{reason}, please provide {current_field.question}"
                    else:
                        # Avoid the confusing "doesn't match the question" message for first interactions
                        if is_first_interaction:
                            message = f"To get started, {current_field.question}"
                        else:
                            message = f"{reason}, {current_field.question}"
                else:
                    # Validation passed but extraction failed - ask the question again
                    message = f"I didn't catch that clearly. {current_field.question}"
                
                yield self.create_text_message(message)
                
                yield self.create_json_message(
                    {
                        "execution_metadata": {
                            LogMetadata.TOTAL_PRICE: self.llm_usage["usage"].total_price
                            if self.llm_usage["usage"] is not None
                            else 0,
                            LogMetadata.CURRENCY: self.llm_usage["usage"].currency
                            if self.llm_usage["usage"] is not None
                            else "",
                            LogMetadata.TOTAL_TOKENS: self.llm_usage["usage"].total_tokens
                            if self.llm_usage["usage"] is not None
                            else 0,
                        }
                    }
                )

                finish_log_data = {
                    "output": message,
                    "dialogue_state": {
                        "completed": False,
                        "current_field_index": self.dialogue_state.current_field_index,
                        "total_fields": len(self.dialogue_state.fields),
                        "completed_fields": [
                            {"name": f.name, "value": f.value}
                            for f in self.dialogue_state.fields
                            if f.value is not None
                        ],
                        "current_field": {
                            "name": current_field.name,
                            "question": current_field.question
                        }
                    },
                    "model_interactions_summary": {
                        "validation_result": is_valid,
                        "extracted_answers": extracted_answers,
                        "total_interactions": 2
                    }
                }
                finish_log_metadata = {
                    LogMetadata.STARTED_AT: round_started_at,
                    LogMetadata.FINISHED_AT: time.perf_counter(),
                    LogMetadata.ELAPSED_TIME: time.perf_counter() - round_started_at,
                    LogMetadata.TOTAL_TOKENS: self.llm_usage["usage"].total_tokens if self.llm_usage["usage"] else 0,
                }

                yield self.finish_log_message(
                    log=round_log,
                    data=finish_log_data,
                    metadata=finish_log_metadata
                )
                self._save_dialogue_state(params.storage_key)
                return

        # Handle potential direct field modification input after user was asked which field to modify
        if not self.dialogue_state.confirmation_requested and not self.dialogue_state.completed and all(f.value for f in self.dialogue_state.fields):
            # Expect patterns like "email: new@example.com" or "phone 123456" etc.
            parts = params.query.strip().split(None, 1)
            delimiter_index = params.query.find(":")
            updated = False
            field_name = None
            new_value = None
            if delimiter_index != -1:
                field_name = params.query[:delimiter_index].strip()
                new_value = params.query[delimiter_index+1:].strip()
            elif len(parts) == 2:
                field_name, new_value = parts[0].strip(), parts[1].strip()
            if field_name and new_value:
                for f in self.dialogue_state.fields:
                    if f.name.lower() == field_name.lower():
                        f.value = new_value
                        self.collected_data[f.name] = new_value
                        updated = True
                        break
            if updated:
                # Ask for confirmation again with updated summary
                self.dialogue_state.confirmation_requested = True
                summary = self._generate_summary()
                message = (
                    "Updated. Please review the collected information. Reply 'yes' to confirm, or specify another change.\n" + summary
                )
                yield self.create_text_message(message)
                self._save_dialogue_state(params.storage_key)
                yield self.create_json_message({
                    "execution_metadata": {
                        LogMetadata.TOTAL_PRICE: self.llm_usage["usage"].total_price if self.llm_usage["usage"] is not None else 0,
                        LogMetadata.CURRENCY: self.llm_usage["usage"].currency if self.llm_usage["usage"] is not None else "",
                        LogMetadata.TOTAL_TOKENS: self.llm_usage["usage"].total_tokens if self.llm_usage["usage"] is not None else 0,
                    }
                })
                finish_log_data = {
                    "output": message,
                    "dialogue_state": {
                        "completed": False,
                        "confirmation_requested": True,
                        "confirmed": False,
                        "current_field_index": self.dialogue_state.current_field_index,
                        "total_fields": len(self.dialogue_state.fields),
                        "completed_fields": [
                            {"name": f.name, "value": f.value}
                            for f in self.dialogue_state.fields
                        ]
                    }
                }
                finish_log_metadata = {
                    LogMetadata.STARTED_AT: round_started_at,
                    LogMetadata.FINISHED_AT: time.perf_counter(),
                    LogMetadata.ELAPSED_TIME: time.perf_counter() - round_started_at,
                    LogMetadata.TOTAL_TOKENS: self.llm_usage["usage"].total_tokens if self.llm_usage["usage"] else 0,
                }
                yield self.finish_log_message(
                    log=round_log,
                    data=finish_log_data,
                    metadata=finish_log_metadata
                )
                return

    def _is_valid_answer(self, model_response: str) -> bool:
        try:
            if model_response.startswith("INVALID:"):
                return False

            return True

        except Exception:
            return False

    def _generate_summary(self) -> str:
        summary = ""
        for field in self.dialogue_state.fields:
            summary += f"{field.question} {field.value}\n"
        return summary.strip()
