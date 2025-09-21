from langchain_openai import AzureChatOpenAI
from backend.core.config import (
    AZURE_OPENAI_ENDPOINT,
    CHAT_DEPLOYMENT_NAME,
    CODE_DEPLOYMENT_NAME,
    OPENAI_API_VERSION,
)

# Lazy construction could be added if needed; keep simple for now.
llm_chat = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=CHAT_DEPLOYMENT_NAME,
    api_version=OPENAI_API_VERSION,
)

llm_code = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=CODE_DEPLOYMENT_NAME,
    api_version=OPENAI_API_VERSION,
)

__all__ = ["llm_chat", "llm_code"]
