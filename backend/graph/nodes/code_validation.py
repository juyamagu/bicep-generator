import os
import shutil
import subprocess
import tempfile
from backend.core.state import State
from backend.domain import Phase, DEFAULT_LANGUAGE
from backend.i18n.messages import get_message
from backend.domain.lint_parser import parse_bicep_lint_output
from backend.core.config import MAX_GENERATION_COUNT
from backend.core.logging import logger


async def code_validation(state: State):
    """Validate the generated Bicep code using the Bicep CLI lint command."""
    logger.debug("state: %s", state)

    language = state.get("language", DEFAULT_LANGUAGE)

    def _get_bicep_command() -> str:
        if shutil.which("az") is not None:
            return "az bicep"
        return "bicep"

    code = state.get("bicep_code", "")
    if not code:
        return {
            "messages": [{"role": "assistant", "content": get_message("messages.code_validation.no_code", language)}],
            "lint_messages": [],
            "validation_passed": False,
        }
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bicep", mode="w", encoding="utf-8") as f:
            f.write(code)
            tmp_path = f.name
        try:
            bicep_cmd = _get_bicep_command()
            proc = subprocess.run(
                " ".join([bicep_cmd, "lint", "--file", tmp_path]),
                capture_output=True,
                text=True,
                timeout=60,
                shell=True,
            )
            lint_output_raw = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
            lint_messages = parse_bicep_lint_output(lint_output_raw)
            logger.debug("lint_messages: %s", lint_messages)

            # Validation logic
            validation_passed = len(lint_messages) == 0
            logger.debug("validation_passed: %s", validation_passed)

            MAX_LINT_MESSAGES_FOR_PREVIEW = 10
            preview_lines = [str(msg) for msg in lint_messages[:MAX_LINT_MESSAGES_FOR_PREVIEW]]
            if len(lint_messages) > MAX_LINT_MESSAGES_FOR_PREVIEW:
                preview_lines.append(f"... ({len(lint_messages)-MAX_LINT_MESSAGES_FOR_PREVIEW} more)")
            lint_output = "\n".join(preview_lines)
            if validation_passed:
                chat_message = get_message("messages.code_validation.lint_succeeded", language)
            else:
                chat_message = get_message("messages.code_validation.lint_failed", language, lint_output=lint_output)
        except FileNotFoundError:
            chat_message = get_message("messages.code_validation.bicep_cmd_not_found", language)
            lint_messages = []
            validation_passed = False
        except Exception as e:  # noqa
            chat_message = get_message("messages.code_validation.execution_error", language, error=str(e))
            lint_messages = []
            validation_passed = False
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    generation_count = state.get("generation_count", 0)
    if not validation_passed and generation_count < MAX_GENERATION_COUNT:
        next_phase = Phase.CODE_GENERATING.value
    else:
        next_phase = Phase.COMPLETED.value
    return {
        "messages": [{"role": "assistant", "content": chat_message}],
        "lint_messages": lint_messages,
        "validation_passed": validation_passed,
        "phase": next_phase,
    }


__all__ = ["code_validation"]
