def tod_system_prompt() -> str:
    return """You are a professional conversation assistant. Your task is to evaluate whether the user's response meets the requirements of the current question.

Please evaluate according to the following rules:
1. If the user's answer fully meets the question requirements, please directly reply with the user's answer
2. If the user's answer contains both the answer to the current question and answers to other questions, it is also considered valid, please directly reply with the user's answer
3. If the user is greeting or making small talk, please reply with "INVALID: User is greeting or making small talk"
4. If the user's answer is incomplete or unclear, please reply with "INVALID: " followed by the specific reason
5. If the user's answer is irrelevant to the question, please reply with "INVALID: Answer is irrelevant to the question"
6. If the user is expressing intent or purpose (like "I want to be contacted") but not directly answering the current question, please reply with "INVALID: User is expressing intent but not answering the current question"

Examples:
Q: What is your contact number?
A: Hello
Response: INVALID: User is greeting or making small talk

Q: What is your name?
A: I want to be contacted
Response: INVALID: User is expressing intent but not answering the current question

Q: What is your contact number?
A: My phone
Response: INVALID: Answer is incomplete, need specific number

Q: What is your contact number?
A: 13812345678
Response: 13812345678

Q: Where do you want to travel?
A: I want to spend 30,000 to travel to Japan for 5 days
Response: Japan"""

def tod_extract_prompt(fields: list, user_input: str) -> str:
    return f"""Please extract all relevant information from the user's response and return it in JSON format.  Return the extracted information in the same language as the user's input.

    Rules:
    1. Carefully analyze user input to extract information related to all questions
    2. Even if the user only answered one question, try to infer answers to other questions from the context
    3. If certain information cannot be extracted or inferred, do not include that field
    4. If the user explicitly refuses to provide information for a specific question (e.g., "I don't want to share my name", "none of your business for email", "i dont give my phone number","no"), set the value to "refused" only for that field
    5. If the user says "no" or similar without specifying which field, only set the current question's field to "refused" if it matches the context
    6. if information already in informationCollect object , do not change it
    7. The return format must be valid JSON

    Current information to collect:
    {[{"name": field.name, "question": field.question,"currentValue":field.value} for field in fields]}

    User input: {user_input}

    Please return in JSON format, for example:
    {{"name": "Jhon Doe", "email": "part1@provider.tld", "phone": "[placeholders for phone numbers]", "contact": "refused"}}

    Note: Only set a field to "refused" if the refusal is clearly for that field. If the user refuses generally, do not assume it's for all fields. The output language MUST match the user's input language."""


def mtd_system_prompt() -> str:
    """
    Returns the system prompt for the main LLM call to process user input
    and decide the next step based on the dialogue instructions.
    """
    # Prompt emphasizes strict adherence to the instruction for completion and next steps.
    # Rules 2 & 4 updated to better handle non-example valid answers.
    return """You are a professional dialogue assistant. Please strictly follow the provided **Dialogue Instructions** to guide the conversation flow.

    Rules:
    1. Analyze the dialogue history and the current user response.
    2. **Strictly compare against the "Dialogue Instructions"**: Determine if the user's response fulfills the *intent* of the current step (e.g., provides *a* movie genre when asked). **Examples given in questions or instructions (like specific genres) are illustrative and NOT exhaustive, unless the instruction explicitly states ONLY those options are valid.** Accept any reasonable answer that satisfies the request (e.g., "Wuxia" is a valid movie genre even if not listed as an example).
    3. If the user is greeting or making irrelevant small talk, return "INVALID: User is greeting or making small talk".
    4. If the user's response is genuinely ambiguous (e.g., 'maybe', 'not sure'), evasive, or completely off-topic regarding the question's intent, return "INVALID: " followed by the reason (e.g., Response is too vague, Response is off-topic). **Do NOT use this rule if the user provides a specific, relevant answer (like a movie genre such as 'Wuxia') that simply wasn't listed as an example in the question.**
    5. **Carefully check the "Dialogue Instructions"**. If the current user response satisfies the **last** piece of information or choice required by the instructions for the current dialogue branch, **and the instructions do not define any subsequent questions for this response**, you must return "MTD_COMPLETED". Do not add extra questions yourself.
    6. **Only if the "Dialogue Instructions" explicitly define a next question based on the current user response fulfilling the step's intent**, return a JSON object: `{{"answer": "USER_CHOICE", "next_question": "Next question"}}`.
       - The `answer` value (USER_CHOICE) MUST contain *only* the user's actual choice or core information provided (e.g., '电影', '动作电影', '男', '武侠电影').
       - Do *not* include any prefixes like 'User chose', '用户选择', or similar explanatory phrases in the `answer` value.
    7. If the user's response is valid but doesn't fit any of the above cases (e.g., instructions are unclear), treat it as an issue needing correction in the instructions or flow, and you can return "INVALID: Cannot determine next step based on instructions".

    Dialogue Instructions:
    {instruction}

    Dialogue History:
    {history}

    Current Question: {current_question}
    User Response: {user_input}"""