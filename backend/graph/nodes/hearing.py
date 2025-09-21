from backend.core.llm import llm_chat
from backend.core.state import State
from backend.domain import Phase, DEFAULT_LANGUAGE
from backend.i18n.messages import get_message
from backend.core.config import MAX_HEARING_COUNT
from backend.core.logging import logger
from backend.graph.helpers import make_dialog_history, to_text


async def hearing(state: State):
    """Conduct the hearing phase to gather user requirements."""
    logger.debug("state: %s", state)

    # Determine if the requirements are sufficient
    language = state.get("language", DEFAULT_LANGUAGE)
    dialog_history = make_dialog_history(state)
    user_content = get_message("prompts.requirements_evaluation.user_instruction", language)
    user_content += f"\n\n## Context:\n{dialog_history}\n"
    messages = [
        {"role": "system", "content": get_message("prompts.requirements_evaluation.system_instruction", language)},
        {"role": "user", "content": user_content},
    ]
    is_sufficient = True
    if state.get("hearing_count", 1) >= MAX_HEARING_COUNT:
        is_sufficient = True
        logger.debug("Reached MAX_HEARING_COUNT, forcing to sufficient")
    else:
        try:
            logger.debug("Invoking LLM with messages: %s", messages)
            resp = llm_chat.invoke(messages)
            logger.debug("Response from LLM: %s", resp.content)

            is_sufficient = "yes" in to_text(resp.content).strip().lower()
            logger.debug("is_sufficient: %s", is_sufficient)
        except Exception as e:  # noqa
            logger.error("Error occurred while invoking LLM: %s", e)

    if is_sufficient:
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": get_message("messages.requirements_evaluation.requirements_are_sufficient", language),
                }
            ],
            "hearing_count": state.get("hearing_count", 1),
            "phase": Phase.SUMMARIZING.value,
        }

    # Ask for more information
    messages = [
        {"role": "system", "content": get_message("prompts.hearing.system_instruction", language)},
        {
            "role": "user",
            "content": get_message("prompts.hearing.user_instruction", language)
            + f"\n\n## Context:\n{dialog_history}\n",
        },
    ]
    logger.debug("Invoking LLM with messages: %s", messages)
    resp = await llm_chat.ainvoke(messages)
    logger.debug("Response from LLM: %s", resp.content)

    question = to_text(resp.content).strip() or get_message("prompts.hearing.fallback_question", language)
    return {
        "messages": [{"role": "assistant", "content": question}],
        "hearing_count": state.get("hearing_count", 1) + 1,
        "phase": Phase.HEARING.value,
    }


def should_hear_again(state: State) -> str:
    if state.get("phase", "") == Phase.HEARING.value:
        return "yes"
    return "no"


__all__ = ["hearing", "should_hear_again"]
