from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama


MODEL_ALIASES = {
    "gpt-4o": "gpt-4o",
    "gpt-3.5-turbo": "gpt-3.5-turbo",
}


def get_llm(model_name: str, temperature: float = 0.0):
    """
    Unified LLM factory with alias handling.
    This makes the backend robust to UI / config variations.
    """

    normalized_model = MODEL_ALIASES.get(model_name, model_name)

    if normalized_model.startswith("gpt"):
        return ChatOpenAI(
            model=normalized_model,
            temperature=temperature
        )

    raise ValueError(
        f"Unsupported model: {model_name}. "
        f"Resolved as: {normalized_model}"
    )
