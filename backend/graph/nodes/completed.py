from backend.core.state import State
from backend.core.config import MAX_GENERATION_COUNT
from backend.core.logging import logger
from backend.domain import DEFAULT_LANGUAGE
from backend.i18n.messages import get_message


async def completed(state: State):
    logger.debug("state: %s", state)

    language = state.get("language", DEFAULT_LANGUAGE)
    generation_count = state.get("generation_count", 0)
    validation_passed = state.get("validation_passed", False)
    chat_message = get_message("messages.completed.default", language)
    if validation_passed:
        chat_message = get_message("messages.completed.validation_passed", language)
    elif generation_count >= MAX_GENERATION_COUNT:
        chat_message = get_message("messages.completed.generation_limit_reached", language)
    return {"messages": [{"role": "assistant", "content": chat_message}]}


__all__ = ["completed"]
