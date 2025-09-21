from backend.core.llm import llm_chat
from backend.core.state import State
from backend.core.logging import logger
from backend.domain import Phase, DEFAULT_LANGUAGE
from backend.i18n.messages import get_message
from backend.graph.helpers import make_dialog_history, to_text


async def summarizing(state: State):
    """Summarize the requirements gathered during the hearing phase."""
    logger.debug("state: %s", state)

    # Summarize the user requirements
    language = state.get("language", DEFAULT_LANGUAGE)
    MAX_DIALOG_HISTORY = 8000
    dialog_history = make_dialog_history(state)
    if len(dialog_history) > MAX_DIALOG_HISTORY:
        dialog_history = dialog_history[-MAX_DIALOG_HISTORY:]
    messages = [
        {"role": "system", "content": get_message("prompts.summarize_requirements.system_instruction", language)},
        {
            "role": "user",
            "content": get_message(
                "prompts.summarize_requirements.user_instruction", language, dialog_history=dialog_history
            ),
        },
    ]
    logger.debug("Invoking LLM with messages: %s", messages)
    resp = await llm_chat.ainvoke(messages)
    logger.debug("Response from LLM: %s", resp.content)

    summary_text = to_text(resp.content).strip()
    if summary_text:
        message = get_message(
            "messages.summarize_requirements.generation_succeeded", language, summary_text=summary_text
        )
    else:
        summary_text = dialog_history
        message = get_message("messages.summarize_requirements.generation_failed", language, summary_text=summary_text)

    return {
        "messages": [{"role": "assistant", "content": message}],
        "requirement_summary": summary_text,
        "phase": Phase.CODE_GENERATING.value,
    }


__all__ = ["summarizing"]
