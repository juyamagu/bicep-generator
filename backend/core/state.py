from typing import List, Dict, Any, Annotated, Optional, TypedDict
from pydantic import BaseModel
from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph.message import add_messages
from backend.domain import Phase, DEFAULT_LANGUAGE
from backend.i18n.messages import get_message
from backend.core.config import MAX_HEARING_COUNT, MAX_GENERATION_COUNT


class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    hearing_count: int
    current_user_message: str
    bicep_code: str
    lint_messages: List[Any]
    validation_passed: bool
    generation_count: int
    phase: str
    requirement_summary: str
    language: str


class ChatMessage(BaseModel):
    content: str
    sender: str
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    session_id: Optional[str] = "default"
    message: Optional[str] = None
    language: Optional[str] = DEFAULT_LANGUAGE
    conversation_history: List[ChatMessage] = []


class ChatResponse(BaseModel):
    message: str
    bicep_code: str = ""
    phase: str = ""
    requirement_summary: str = ""
    requires_user_input: bool = True


def initial_state(language: str = DEFAULT_LANGUAGE) -> Dict[str, Any]:
    return {
        "messages": [AIMessage(role="assistant", content=get_message("messages.hearing.initial", language))],
        "hearing_count": 1,
        "current_user_message": "",
        "bicep_code": "",
        "lint_messages": [],
        "validation_passed": False,
        "generation_count": 0,
        "phase": Phase.HEARING.value,
        "requirement_summary": "",
        "language": language,
    }


# ensure_session_initialized will be placed in graph.graph to avoid circular import
__all__ = [
    "State",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "initial_state",
    "MAX_HEARING_COUNT",
    "MAX_GENERATION_COUNT",
]
