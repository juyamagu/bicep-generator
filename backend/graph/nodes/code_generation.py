import re
from typing import List
from backend.core.llm import llm_code
from backend.core.state import State
from backend.core.config import MAX_BICEP_CODE_LENGTH_FOR_CONTEXT, MAX_LINT_MESSAGES_FOR_CONTEXT
from backend.domain import Phase, DEFAULT_LANGUAGE
from backend.domain.bicep_parser import parse_bicep_blocks, BicepCodeBlock
from backend.domain.lint_parser import BicepLintMessage
from backend.domain.utils import fetch_url_content
from backend.i18n.messages import get_message
from backend.core.logging import logger
from backend.graph.helpers import to_text


async def code_generation(state: State):
    """Generate or regenerate Bicep code based on the summarized requirements and any lint messages."""
    logger.debug("state: %s", state)

    language = state.get("language", DEFAULT_LANGUAGE)
    requirement_summary = state.get("requirement_summary")
    assert isinstance(requirement_summary, str) and requirement_summary.strip(), "requirement_summary is required"
    lint_messages = state.get("lint_messages", [])
    bicep_code = state.get("bicep_code", "")

    # Determine if this is an initial generation or a regeneration
    generation_count = state.get("generation_count", 0)
    if generation_count == 0:
        system_prompt = get_message("prompts.code_generation.system_initial", language)
        user_prompt = get_message(
            "prompts.code_generation.user_initial", language, requirement_summary=requirement_summary
        )
        chat_message = get_message("messages.code_generation.initial_done", language)
    else:
        system_prompt = get_message("prompts.code_generation.system_regeneration", language)
        user_prompt = get_message(
            "prompts.code_generation.user_regeneration", language, requirement_summary=requirement_summary
        )
        chat_message = get_message("messages.code_generation.regeneration_done", language)

    # Append context from lint messages and existing Bicep code
    context = _retrieve_code_generation_context(lint_messages, bicep_code, llm_code=llm_code)
    user_prompt += context
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    try:
        logger.debug("Invoking LLM with messages: %s", messages)
        resp = await llm_code.ainvoke(messages)
        logger.debug("Response from LLM: %s", resp.content)

        new_bicep_code = to_text(resp.content).strip()
        m = re.search(r"```(?:bicep)?\s*\n([\s\S]*?)```", new_bicep_code, flags=re.IGNORECASE)
        if m:
            new_bicep_code = m.group(1).strip()
        return {
            "messages": [{"role": "assistant", "content": chat_message}],
            "bicep_code": new_bicep_code,
            "generation_count": generation_count + 1,
            "phase": Phase.CODE_VALIDATING.value,
        }
    except Exception as e:  # noqa
        logger.error("Error occurred: %s", e)
        error_message = get_message("messages.code_generation.generation_failed", language, error=str(e))
        return {
            "messages": [{"role": "assistant", "content": error_message}],
            "bicep_code": bicep_code,
            "generation_count": generation_count + 1,
            "phase": Phase.CODE_GENERATING.value,
        }


def _ask_llm_for_guidance(lint_message: BicepLintMessage, code_blocks: List[BicepCodeBlock]) -> str:
    """Ask the LLM for guidance on how to fix a specific lint message."""

    # Find the relevant code block for the lint message
    line = lint_message.line
    block = None
    for b in code_blocks:
        if b.start_line <= line <= b.end_line:
            block = b
            break

    if not block:
        logger.debug("No code block found for lint messages")
        return ""

    m = re.match(r"resource\s+(\w+)\s+'([^@]+)@([^']+)'", block.text)
    if not m:
        logger.debug("Unsupported resource declaration: %s", block)
        return ""
    resource_type = m.group(2)

    # Fetch documents
    docs_url = (
        f"https://learn.microsoft.com/en-us/azure/templates/{resource_type.lower()}/?pivots=deployment-language-bicep"
    )
    docs_content = fetch_url_content(
        docs_url,
        timeout=5.0,
        max_bytes=1_000_000,
        query_selector='div[data-pivot="deployment-language-bicep"]',
        compressed=True,
    )
    if not docs_content:
        logger.debug("Failed to fetch docs content from %s", docs_url)
        return ""

    # Ask LLM for guidance
    guidance = ""
    messages = [
        {
            "role": "system",
            "content": "You are an expert in Bicep. Provide concise guidance to fix an issue.",
        },
        {
            "role": "user",
            "content": "\n\n".join(
                [
                    "How can I fix the following issue in the Bicep code?",
                    f"## Lint message\n{str(lint_message)}",
                    f"## Bicep code block\n- start: {block.start_line}\n- end: {block.end_line}\n\n```bicep{block.text}\n```",
                    f"## Learn doc:\n{docs_content}\n",
                ]
            ),
        },
    ]
    try:
        logger.debug("Invoking LLM with messages: %s", messages)
        # TODO: llm_chat or llm_code?
        resp = llm_code.invoke(messages)
        logger.debug("Response from LLM: %s", resp.content)
        guidance = to_text(resp.content).strip()
    except Exception as e:
        logger.error("Error occurred while invoking LLM for guidance: %s", e)

    if guidance:
        return guidance

    return ""


def _retrieve_code_generation_context(lint_messages: List[BicepLintMessage], bicep_code: str, llm_code=None) -> str:
    """Retrieve context for code generation from lint messages and existing Bicep code."""
    bicep_code_for_context = (
        bicep_code
        if len(bicep_code) <= MAX_BICEP_CODE_LENGTH_FOR_CONTEXT
        else bicep_code[:MAX_BICEP_CODE_LENGTH_FOR_CONTEXT] + "\n(... truncated)"
    )

    if not lint_messages and not bicep_code:
        return ""
    if bicep_code and not lint_messages:
        return f"## Bicep Code\n{bicep_code_for_context}"
    if lint_messages and not bicep_code:
        return "## Lint Messages\n" + "\n".join([str(m) for m in lint_messages])

    lines = []
    code_blocks = parse_bicep_blocks(bicep_code)
    for lint_message in lint_messages[:MAX_LINT_MESSAGES_FOR_CONTEXT]:
        guidance = _ask_llm_for_guidance(lint_message, code_blocks)
        line = "\n\n".join(
            [
                "### Lint Message",
                str(lint_message),
                "### Guidance",
                guidance or "N/A",
            ]
        )
        lines.append(line)

    return "\n\n".join(["## Bicep Code", bicep_code_for_context, "## Lint Messages and Guidance", "\n\n".join(lines)])


__all__ = ["code_generation"]
