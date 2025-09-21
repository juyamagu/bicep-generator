import os
from typing import Optional
import dotenv

dotenv.load_dotenv()

AZURE_OPENAI_ENDPOINT: Optional[str] = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_VERSION: Optional[str] = os.getenv("OPENAI_API_VERSION")
CHAT_DEPLOYMENT_NAME: str = os.getenv("CHAT_DEPLOYMENT_NAME", "gpt-4.1")
CODE_DEPLOYMENT_NAME: str = os.getenv("CODE_DEPLOYMENT_NAME", "gpt-4.1")

DEBUG_LOG: bool = os.getenv("DEBUG_LOG", "0") in ("1", "true", "True")
MAX_HEARING_COUNT: int = int(os.getenv("MAX_HEARING_COUNT", "20"))
MAX_GENERATION_COUNT: int = int(os.getenv("MAX_GENERATION_COUNT", "5"))

MAX_LINT_MESSAGES_FOR_CONTEXT: int = int(os.getenv("MAX_LINT_MESSAGES_FOR_CONTEXT", "3"))
MAX_BICEP_CODE_LENGTH_FOR_CONTEXT: int = int(os.getenv("MAX_BICEP_CODE_LENGTH_FOR_CONTEXT", "8000"))

__all__ = [
    "AZURE_OPENAI_ENDPOINT",
    "OPENAI_API_VERSION",
    "CHAT_DEPLOYMENT_NAME",
    "CODE_DEPLOYMENT_NAME",
    "DEBUG_LOG",
    "MAX_HEARING_COUNT",
    "MAX_GENERATION_COUNT",
]
