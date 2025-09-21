import os
import re
import shutil
import tempfile
import subprocess
import logging
from datetime import datetime
from typing import List, Dict, Any, Annotated, TypedDict, Optional

import dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages, AnyMessage
from langchain_core.messages import AIMessage

try:
    from .constants import Phase, REQUIRE_USER_INPUT_PHASES, DEFAULT_LANGUAGE  # type: ignore
    from .i18n.messages import get_message  # type: ignore
    from .lint_parser import parse_bicep_lint_output, BicepLintMessage  # type: ignore
    from .bicep_parser import parse_bicep_blocks  # type: ignore
    from .utils import fetch_url_content  # type: ignore
except ImportError:
    from constants import Phase, REQUIRE_USER_INPUT_PHASES, DEFAULT_LANGUAGE  # type: ignore
    from i18n.messages import get_message  # type: ignore
    from lint_parser import parse_bicep_lint_output, BicepLintMessage  # type: ignore
    from bicep_parser import parse_bicep_blocks  # type: ignore
    from utils import fetch_url_content  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
# Checkpointer: Automatically fall back to MemorySaver when SqliteSaver is not available
# ──────────────────────────────────────────────────────────────────────────────
SqliteSaver = None
MemorySaver = None
try:
    from langgraph.checkpoint.sqlite import SqliteSaver as _SqliteSaver  # type: ignore

    SqliteSaver = _SqliteSaver
except Exception:
    try:
        from langgraph.checkpoint.memory import MemorySaver as _MemorySaver

        MemorySaver = _MemorySaver
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────────────────────
# Environment variables
# ──────────────────────────────────────────────────────────────────────────────
dotenv.load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
CHAT_DEPLOYMENT_NAME = os.getenv("CHAT_DEPLOYMENT_NAME", "gpt-4.1")
CODE_DEPLOYMENT_NAME = os.getenv("CODE_DEPLOYMENT_NAME", "gpt-4.1")

DEBUG_LOG = os.getenv("DEBUG_LOG", "0") in ("1", "true", "True")  # Enable debug logging
MAX_HEARING_COUNT = int(os.getenv("MAX_HEARING_COUNT", "20"))  # Maximum number of hearing attempts
MAX_GENERATION_COUNT = int(os.getenv("MAX_GENERATION_COUNT", "5"))  # Maximum number of code generation attempts


# ──────────────────────────────────────────────────────────────────────────────
# Logging configuration
# ──────────────────────────────────────────────────────────────────────────────
# Console: controlled by DEBUG_LOG
# Log file: always DEBUG level

_LOG_LEVEL = logging.DEBUG if DEBUG_LOG else logging.INFO
logger = logging.getLogger("bicep-generator")
logger.setLevel(logging.DEBUG)
_has_console = any(
    isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers
)
if not _has_console:
    _console_handler = logging.StreamHandler()
    _console_handler.setLevel(_LOG_LEVEL)
    _console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(funcName)s] %(message)s"))
    logger.addHandler(_console_handler)
else:
    for _h in logger.handlers:
        if isinstance(_h, logging.StreamHandler) and not isinstance(_h, logging.FileHandler):
            _h.setLevel(_LOG_LEVEL)

_log_filename = datetime.now().strftime("%Y%m%d.log")
_log_file_path = os.getenv("APP_LOG_FILE", os.path.join(os.path.dirname(__file__), "logs", _log_filename))
try:
    os.makedirs(os.path.dirname(_log_file_path), exist_ok=True)
except Exception:
    pass

if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    _file_handler = logging.FileHandler(_log_file_path, encoding="utf-8")
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s:%(funcName)s] %(message)s"))
    logger.addHandler(_file_handler)

logger.propagate = False

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI initialization & CORS
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Bicep Generator API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# LLM clients
# ──────────────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
# State / I/O model
# ──────────────────────────────────────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    hearing_count: int
    current_user_message: str
    bicep_code: str  # Most recently generated code
    lint_messages: List[BicepLintMessage]  # Parsed lint results
    validation_passed: bool  # Whether lint validation passed
    generation_count: int  # Number of generation attempts
    phase: str  # Next phase name (visible to frontend)
    requirement_summary: str  # Requirement summary for code_generation prompt
    language: str  # Language setting ("ja" | "en")


class ChatMessage(BaseModel):
    content: str
    sender: str
    timestamp: Optional[str] = None  # Optional; may be omitted if unused


class ChatRequest(BaseModel):
    session_id: Optional[str] = "default"  # Recommended to provide conversation ID from the frontend
    message: Optional[str] = None  # If empty, proceed only with the AI's next step
    language: Optional[str] = DEFAULT_LANGUAGE  # Language setting ("ja" | "en")
    conversation_history: List[ChatMessage] = []  # Unused (storage is handled by the checkpointer)


class ChatResponse(BaseModel):
    message: str
    bicep_code: str = ""
    phase: str = ""
    requirement_summary: str = ""
    requires_user_input: bool = True  # Whether the frontend must wait for the next user input (True = wait)


def initial_state(language: str = DEFAULT_LANGUAGE) -> Dict[str, Any]:
    """Return a fresh initial state dict. Use this for both first-time initialization and resets."""
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


def ensure_session_initialized(config: Dict[str, Any], language: str = DEFAULT_LANGUAGE) -> None:
    """Ensure the session identified by config has a full initial state

    If the stored state is missing or incomplete, overwrite it with a fresh initial state
    (preserving the requested language).
    """
    try:
        existing = GRAPH.get_state(config).values  # type: ignore[arg-type]
        # Consider session uninitialized when 'phase' is missing or empty
        if not existing or not existing.get("phase"):
            GRAPH.update_state(config, initial_state(language))  # type: ignore[arg-type]
    except Exception:
        # If anything goes wrong (e.g., missing key), try to initialize anyway
        try:
            GRAPH.update_state(config, initial_state(language))  # type: ignore[arg-type]
        except Exception:
            # If update fails, allow caller to proceed; errors will surface later
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Node definitions (prompts are left unchanged)
# ──────────────────────────────────────────────────────────────────────────────
def _to_text(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: List[str] = []
        for r in raw:
            if isinstance(r, str):
                parts.append(r)
            elif isinstance(r, dict):
                # Common OpenAI message part patterns
                val = r.get("text") or r.get("content") or ""
                if isinstance(val, list):
                    parts.append(" ".join(str(v) for v in val))
                else:
                    parts.append(str(val))
            else:
                parts.append(str(r))
        return "\n".join(p for p in parts if p)
    return str(raw)


def _make_dialog_history(state: State) -> str:
    """Generate dialog history from State for LLM invocation

    The dialog history is stored in state["messages"] and must be converted appropriately.
    The returned format is a string where each message is prefixed by its role and content.
    """
    dialog = ""
    for msg in state.get("messages", []):
        role = None
        content_val = None
        if hasattr(msg, "type") and hasattr(msg, "content"):
            if msg.type == "human":
                role = "user"
            elif msg.type == "ai":
                role = "assistant"
            content_val = getattr(msg, "content", "")
        elif isinstance(msg, dict):
            role = str(msg.get("role"))
            content_val = msg.get("content")
        elif isinstance(msg, str):
            role = "assistant"
            content_val = msg
        if role and content_val is not None:
            dialog += f"[{role.upper()}]: {content_val}\n"
    return dialog


async def hearing(state: State):
    """Requirement-hearing node

    Determine whether to continue the hearing; if necessary, ask a single short follow-up question.
    """
    language = state.get("language", DEFAULT_LANGUAGE)
    dialog_history = _make_dialog_history(state)

    # Determine whether to continue the hearing
    user_content = get_message("prompts.requirements_evaluation.user_instruction", language)
    user_content += f"\n\n## Context:\n{dialog_history}\n"
    messages = [
        {
            "role": "system",
            "content": get_message("prompts.requirements_evaluation.system_instruction", language),
        },
        {"role": "user", "content": user_content},
    ]
    is_sufficient = True  # default to sufficient to avoid infinite loop
    if state.get("hearing_count", 1) >= MAX_HEARING_COUNT:
        is_sufficient = True
        logger.debug("Reached MAX_HEARING_COUNT, forcing to sufficient")
    else:
        try:
            resp = llm_chat.invoke(messages)
            is_sufficient = "yes" in _to_text(resp.content).strip().lower()
            logger.debug("is_sufficient: %s", is_sufficient)
        except Exception as e:  # noqa
            logger.error("Error occurred while invoking LLM: %s", e)

    # If no further hearing is necessary, proceed to requirement summarization
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

    # If continuing the hearing, ask a single short follow-up question
    messages = [
        {
            "role": "system",
            "content": get_message("prompts.hearing.system_instruction", language),
        },
        {
            "role": "user",
            "content": get_message("prompts.hearing.user_instruction", language)
            + f"\n\n## Context:\n{dialog_history}\n",
        },
    ]
    logger.debug("Messages to LLM: %s", messages)
    resp = await llm_chat.ainvoke(messages)
    logger.debug("Response from LLM: %s", resp.content)

    question = _to_text(resp.content).strip() or get_message("prompts.hearing.fallback_question", language)
    return {
        "messages": [{"role": "assistant", "content": question}],
        "hearing_count": state.get("hearing_count", 1) + 1,
        "phase": Phase.HEARING.value,
    }


def should_hear_again(state: State) -> str:
    """Determine whether to continue the hearing: returns 'yes' or 'no' (synchronous function)"""
    phase = state.get("phase", "")
    if phase == Phase.HEARING.value:
        return "yes"
    return "no"


async def summarizing(state: State):
    """Requirement summarization node

    Generate a requirement summary from the entire hearing conversation and store it in state.
    """
    language = state.get("language", DEFAULT_LANGUAGE)

    # Serialize dialog history (trim if it's too long)
    MAX_DIALOG_HISTORY = 8000
    dialog_history = _make_dialog_history(state)
    if len(dialog_history) > MAX_DIALOG_HISTORY:
        dialog_history = dialog_history[-MAX_DIALOG_HISTORY:]

    # Requirement summary generation prompt
    messages = [
        {
            "role": "system",
            "content": get_message("prompts.summarize_requirements.system_instruction", language),
        },
        {
            "role": "user",
            "content": get_message(
                "prompts.summarize_requirements.user_instruction", language, dialog_history=dialog_history
            ),
        },
    ]

    # LLM invocation
    logger.debug("Messages to LLM: %s", messages)
    resp = await llm_chat.ainvoke(messages)
    summary_text = _to_text(resp.content).strip()
    logger.debug("Response from LLM: %s", resp.content)

    # If summarization failed, fall back to using the dialog history as the summary
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
        "phase": Phase.CODE_GENERATING.value,  # Next phase: code_generating
    }


def _retrieve_code_generation_context(lint_messages: List[BicepLintMessage], bicep_code: str) -> str:
    """Build context text describing fix points based on lint messages

    Returns a string that contains relevant code and/or lint information to provide context
    for subsequent code generation requests.
    """
    MAX_BICEP_CODE = 8000
    MAX_LINT_MESSAGES_FOR_CONTEXT = 3

    bicep_code_for_context = (
        bicep_code if len(bicep_code) <= MAX_BICEP_CODE else bicep_code[:MAX_BICEP_CODE] + "\n(... truncated)"
    )

    # If both lint messages and code are empty, no context is needed
    if not lint_messages and not bicep_code:
        return ""
    # If there are no lint messages but code exists, return the code as context
    if bicep_code and not lint_messages:
        return f"## Bicep Code\n{bicep_code_for_context}"
    # If there is lint but no code, return the lint messages (this should not normally occur)
    if lint_messages and not bicep_code:
        logger.debug("Lint messages but no Bicep code")
        return "## Lint Messages\n" + "\n".join([str(m) for m in lint_messages])

    context_lines = []
    bicep_code_blocks = parse_bicep_blocks(bicep_code)

    for lint_message in lint_messages[:MAX_LINT_MESSAGES_FOR_CONTEXT]:
        logger.debug("Processing lint message: %s", lint_message)

        line = lint_message.line

        # Search for the block that contains the relevant line
        block = None
        for b in bicep_code_blocks:
            if b.start_line <= line <= b.end_line:
                block = b
                break

        # If no matching code block is found
        if not block:
            logger.debug("No code block found for lint message: %s", lint_message)
            context_lines.append(f"## Lint Message at line {line}\n{str(lint_message)}")
            continue

        # If the block is not a resource
        if not block.kind == "resource":
            logger.debug("Block at line %s is not a resource, kind=%s", line, block.kind)
            context_lines.append(f"## Lint Message at line {line}\n{str(lint_message)}")
            continue

        # For resource blocks: extract provider, type, and API version
        resource_name = ""
        resource_type = ""
        api_version = ""
        m = re.match(r"resource\s+(\w+)\s+'([^@]+)@([^']+)'", block.text)
        if m:
            resource_name = m.group(1)
            resource_type = m.group(2)
            api_version = m.group(3)
            logger.debug(
                "Found resource block for lint message at line %s: %s, %s, %s",
                line,
                resource_name,
                resource_type,
                api_version,
            )

        # If the resource string could not be parsed
        if not resource_type:
            logger.debug("Failed to parse resource block for lint message at line %s", line)
            context_lines.append(f"## Lint Message at line {line}\n{str(lint_message)}")
            continue

        # Fetch the relevant official documentation page
        # Sample: https://learn.microsoft.com/en-us/azure/templates/Microsoft.Network/virtualNetworks/virtualNetworkPeerings?pivots=deployment-language-bicep
        docs_url = f"https://learn.microsoft.com/en-us/azure/templates/{resource_type.lower()}/?pivots=deployment-language-bicep"
        docs_content = fetch_url_content(
            docs_url,
            timeout=5.0,
            max_bytes=1_000_000,
            query_selector='div[data-pivot="deployment-language-bicep"]',
            compressed=True,
        )
        logger.debug("Fetched docs content from %s, %s bytes", docs_url, len(docs_content))

        # Consult the LLM for improvement suggestions based on lint messages, code block, and documentation
        messages = [
            {
                "role": "system",
                "content": "You are an expert in Bicep. Your role is to provide useful guidance for fixing an inappropriate Bicep code based on the lint message and official documentation. Please make sure your suggestions are concise and directly relevant to the lint message. No other information such as next steps or additional comments is needed.",
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
        logger.debug("Calling LLM for guidance")
        resp = llm_code.invoke(messages)
        guidance = _to_text(resp.content).strip()
        logger.debug("Guidance from LLM:\n%s", guidance)
        if not guidance:
            context_lines.append(f"## Lint Message at line {line}\n{str(lint_message)}")
            continue
        context_lines.append(
            f"## Lint Message at line {line}\n### Lint Message\n{str(lint_message)}\n### Guidance\n{guidance}"
        )

    logger.debug("context_lines: %s", context_lines)
    return f"## Bicep Code\n{bicep_code_for_context}" + "\n\n" + "\n\n".join(context_lines)


async def code_generating(state: State):
    """Bicep code generation / regeneration node

    - Generate code based on the requirement summary and recent lint results
    - Use a regeneration prompt if previously generated code exists
    """
    language = state.get("language", DEFAULT_LANGUAGE)
    requirement_summary = state.get("requirement_summary")
    assert (
        isinstance(requirement_summary, str) and requirement_summary.strip()
    ), "requirement_summary is required for code generation"
    lint_messages = state.get("lint_messages", [])
    bicep_code = state.get("bicep_code", "")
    generation_count = state.get("generation_count", 0)

    # Build prompts
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

    # Append additional context
    context = _retrieve_code_generation_context(lint_messages, bicep_code)
    user_prompt += context
    logger.debug("context: %s", context)

    # LLM invocation
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    logger.debug("Calling LLM to generate a Bicep code")
    resp = await llm_code.ainvoke(messages)
    logger.debug("Response from LLM: %s", resp.content)

    # If there's a code block, extract only its contents
    code_text = _to_text(resp.content).strip()
    m = re.search(r"```(?:bicep)?\s*\n([\s\S]*?)```", code_text, flags=re.IGNORECASE)
    if m:
        code_text = m.group(1).strip()
    logger.debug("Generated code: %s", code_text)

    return {
        "messages": [{"role": "assistant", "content": chat_message}],
        "bicep_code": code_text,
        "generation_count": generation_count + 1,
        "phase": Phase.CODE_VALIDATING.value,  # Next phase: code_validating
    }


async def code_validating(state: State):
    """Bicep code validation node

    Save the generated code to a temporary file and validate it using `az bicep lint`.
    Parse the validation output and store it in the state.
    """
    language = state.get("language", DEFAULT_LANGUAGE)
    chat_message = ""

    def _get_bicep_command() -> str:
        if shutil.which("az") is not None:
            return "az bicep"
        return "bicep"

    # Obtain the generated code. The editor may be empty; if so, skip validation and return.
    code = state.get("bicep_code", "")
    if not code:
        return {
            "messages": [{"role": "assistant", "content": get_message("messages.code_validation.no_code", language)}],
            "lint_messages": [],
            "validation_passed": False,
        }

    # Save to a temporary file and run lint
    tmp_path = None
    lint_output = ""
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

            # Parse lint results
            lint_messages = parse_bicep_lint_output(lint_output_raw)
            logger.debug("lint_messages: %s", lint_messages)

            # Determine whether validation passed
            validation_passed = len(lint_messages) == 0

            # Limit the preview to at most 10 items
            MAX_LINT_MESSAGES_FOR_PREVIEW = 10
            preview_lines = [str(lint_msg) for lint_msg in lint_messages[:MAX_LINT_MESSAGES_FOR_PREVIEW]]
            if len(lint_messages) > MAX_LINT_MESSAGES_FOR_PREVIEW:
                preview_lines.append(f"... ({len(lint_messages)-MAX_LINT_MESSAGES_FOR_PREVIEW} more)")
            lint_output = "\n".join(preview_lines)

            # Construct user-facing message
            if validation_passed:
                chat_message = get_message("messages.code_validation.lint_succeeded", language)
            else:
                chat_message = get_message(
                    "messages.code_validation.lint_failed",
                    language,
                    lint_output=lint_output,
                )
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

    # Decide the next phase
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


def should_regenerate_code(state: State) -> str:
    phase = state.get("phase", "")
    if phase == Phase.CODE_GENERATING.value:
        return "yes"
    return "no"


async def completed(state: State):
    """Completion node

    - Reached when either the hearing or code generation flow completes
    - Also reached when the regeneration limit is hit
    - Simply returns the final message
    """
    language = state.get("language", DEFAULT_LANGUAGE)
    generation_count = state.get("generation_count", 0)
    validation_passed = state.get("validation_passed", False)

    chat_message = get_message("messages.completed.default", language)
    if validation_passed:
        chat_message = get_message("messages.completed.validation_passed", language)
    elif generation_count >= MAX_GENERATION_COUNT:
        chat_message = get_message("messages.completed.generation_limit_reached", language)

    return {
        "messages": [{"role": "assistant", "content": chat_message}],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Graph construction (with checkpointer: SQLite -> fall back to MemorySaver if unavailable)
# ──────────────────────────────────────────────────────────────────────────────
def build_graph():
    gb = StateGraph(State)
    gb.add_node(Phase.HEARING.value, hearing)
    gb.add_node(Phase.SUMMARIZING.value, summarizing)
    gb.add_node(Phase.CODE_GENERATING.value, code_generating)
    gb.add_node(Phase.CODE_VALIDATING.value, code_validating)
    gb.add_node(Phase.COMPLETED.value, completed)

    gb.set_entry_point(Phase.HEARING.value)
    # for debug
    # gb.add_edge(Phase.HEARING.value, Phase.SUMMARIZING.value)
    gb.add_conditional_edges(
        Phase.HEARING.value,
        should_hear_again,
        {"yes": Phase.HEARING.value, "no": Phase.SUMMARIZING.value},
    )
    gb.add_edge(Phase.SUMMARIZING.value, Phase.CODE_GENERATING.value)
    gb.add_edge(Phase.CODE_GENERATING.value, Phase.CODE_VALIDATING.value)
    gb.add_conditional_edges(
        Phase.CODE_VALIDATING.value,
        should_regenerate_code,
        {"yes": Phase.CODE_GENERATING.value, "no": Phase.COMPLETED.value},
    )
    gb.set_finish_point(Phase.COMPLETED.value)

    if SqliteSaver is not None:
        checkpointer = SqliteSaver("checkpoints.db")
        if DEBUG_LOG:
            logger.debug("Using SqliteSaver(checkpoints.db)")
    elif MemorySaver is not None:
        checkpointer = MemorySaver()
        if DEBUG_LOG:
            logger.debug("Using MemorySaver (no persistence)")
    else:
        raise RuntimeError("No available checkpointer: SqliteSaver and MemorySaver are both unavailable")

    return gb.compile(checkpointer=checkpointer)


GRAPH = build_graph()


# ──────────────────────────────────────────────────────────────────────────────
# Routing
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "Bicep Generator API is running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "bicep-generator-api"}


@app.get("/config")
async def get_config():
    return {
        "azure_openai_endpoint": AZURE_OPENAI_ENDPOINT,
        "chat_deployment_name": CHAT_DEPLOYMENT_NAME,
        "code_deployment_name": CODE_DEPLOYMENT_NAME,
        "openai_api_version": OPENAI_API_VERSION,
    }


@app.post("/reset")
async def reset_conversation(session_id: Optional[str] = "default"):
    """
    Conversation reset (simple)
    - In production, it is safest for the frontend to assign a UUID per conversation and change the thread_id
    - Here we simply clear messages, bicep_code and related fields
    """
    config = {"configurable": {"thread_id": session_id or "default"}}  # type: ignore[assignment]
    GRAPH.update_state(  # type: ignore[arg-type]
        config,  # type: ignore[arg-type]
        initial_state(),
    )
    return {"message": f"The conversation (session_id: {session_id}) has been reset."}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Usage:
      1) POST /chat {"session_id":"abc","message":"I need storage"}
         -> a question is returned (is_complete=false)
      2) POST /chat {"session_id":"abc","message":"For a web app"}
         -> next question (is_complete=false)
      3) …repeat
      4) When requirements are sufficient, code_generation runs and returns bicep_code + is_complete=true
    """
    try:
        session_id = request.session_id or "default"
        config = {"configurable": {"thread_id": session_id}}  # type: ignore[assignment]
        logger.debug("session_id=%s request.message=%r", session_id, request.message)

        # 1) Add language setting and the human's utterance to the state
        language = request.language or DEFAULT_LANGUAGE
        state_update = {"language": language}
        if request.message:
            state_update["messages"] = [{"role": "user", "content": request.message}]  # type: ignore[assignment]

        # Initialize the session (only if the state is empty)
        ensure_session_initialized(config, language)

        GRAPH.update_state(  # type: ignore[arg-type]
            config,  # type: ignore[arg-type]
            state_update,
        )

        # 2) Advance the graph (take only the first update from the async stream and return immediately)
        state = GRAPH.get_state(config).values  # type: ignore[arg-type]
        async for _chunk in GRAPH.astream(None, config=config, stream_mode="updates"):  # type: ignore[arg-type]
            break

        # 3) Retrieve the current state
        state = GRAPH.get_state(config).values  # type: ignore[arg-type]
        messages = state.get("messages", [])
        bicep_code = state.get("bicep_code") or ""
        phase = state.get("phase", "")
        requirement_summary = state.get("requirement_summary", "")

        # Latest AI utterance
        latest_ai_text = None
        for msg in reversed(messages):
            if isinstance(msg, dict):
                if msg.get("role") == "assistant":
                    latest_ai_text = msg.get("content")
                    break
            elif hasattr(msg, "type") and getattr(msg, "type") == "ai":
                latest_ai_text = getattr(msg, "content", None)
                break
            elif isinstance(msg, str):
                latest_ai_text = msg
                break

        # Determine whether user input is required
        # - True: when waiting for user's answer (mainly during hearing phase)
        # - False: intermediate states where the system should auto-advance
        requires_user_input = phase in {p.value for p in REQUIRE_USER_INPUT_PHASES}
        is_completed = phase == Phase.COMPLETED.value
        if is_completed:
            requires_user_input = False

        # Default message when complete
        default_msg = "Bicep code generation complete!" if is_completed else "Preparing the next question..."

        return ChatResponse(
            message=latest_ai_text or default_msg,
            bicep_code=bicep_code,
            phase=phase,
            requirement_summary=requirement_summary,
            requires_user_input=requires_user_input,
        )

    except Exception as e:  # noqa
        raise HTTPException(status_code=500, detail="Error occurred during processing: " + str(e))


# ──────────────────────────────────────────────────────────────────────────────
# dev run
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
