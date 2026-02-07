from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """
You are a senior research analyst.
Use ONLY the provided context.
Do NOT hallucinate.
If information is missing, say so explicitly.
"""

def get_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}")
    ])
