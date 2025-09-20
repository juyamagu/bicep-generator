import os
import re
import shutil
import tempfile
import subprocess
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
# Checkpointer: SqliteSaver が無い環境では自動で MemorySaver に切替
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
# 環境変数
# ──────────────────────────────────────────────────────────────────────────────
dotenv.load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
CHAT_DEPLOYMENT_NAME = os.getenv("CHAT_DEPLOYMENT_NAME", "gpt-4.1")
CODE_DEPLOYMENT_NAME = os.getenv("CODE_DEPLOYMENT_NAME", "gpt-4.1")

DEBUG_LOG = os.getenv("DEBUG_LOG", "0") in ("1", "true", "True")  # デバッグログ出力
MAX_HEARING_COUNT = int(os.getenv("MAX_HEARING_COUNT", "20"))  # ヒアリングの最大回数
MAX_GENERATION_COUNT = int(os.getenv("MAX_GENERATION_COUNT", "5"))  # コード生成の最大回数

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI 初期化 & CORS
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
# LLM クライアント
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
# State / I/O モデル
# ──────────────────────────────────────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    hearing_count: int
    current_user_message: str
    bicep_code: str  # 最新生成コード
    lint_messages: List[BicepLintMessage]  # 解析済み lint 結果
    validation_passed: bool  # lint 合格
    generation_count: int  # 生成回数
    phase: str  # 次のフェーズ名（フロントで見えるフェーズ）
    requirement_summary: str  # ヒアリング要件サマリ（code_generation プロンプト用）
    language: str  # 言語設定 ("ja" | "en")


class ChatMessage(BaseModel):
    content: str
    sender: str
    timestamp: Optional[str] = None  # 使わない場合は省略可


class ChatRequest(BaseModel):
    session_id: Optional[str] = "default"  # フロントから会話IDを渡すのが推奨
    message: Optional[str] = None  # 空の場合は「AIの次のステップだけ」進める
    language: Optional[str] = DEFAULT_LANGUAGE  # 言語設定 ("ja" | "en")
    conversation_history: List[ChatMessage] = []  # 未使用（保持はcheckpointerに任せる）


class ChatResponse(BaseModel):
    message: str
    bicep_code: str = ""
    phase: str = ""
    requirement_summary: str = ""
    requires_user_input: bool = True  # フロントエンドが次のユーザー入力を待つ必要があるか（True=待つ）


# 初期化ヘルパー: State のデフォルトを返す関数とセッション初期化関数
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
# ノード定義（プロンプトはそのまま）
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
    """State から LLM invoke 用の会話履歴を生成する

    会話履歴は state["messages"] に格納されているが、適切に適切に変換する必要がある。
    返す形式は List[{"role": "user"|"assistant", "content": "..."}]
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
    """要件ヒアリング ノード

    ヒアリングを続けるべきかの判定を先に行い、必要なら 1 問だけ短く投げる"""
    print("[hearing] called")
    print("[state]", state)

    language = state.get("language", DEFAULT_LANGUAGE)
    dialog_history = _make_dialog_history(state)

    # ヒアリングを続けるべきかの判定
    user_content = get_message("prompts.requirements_evaluation.user_instruction", language)
    user_content += f"\n\n## Context:\n{dialog_history}\n"
    messages = [
        {
            "role": "system",
            "content": get_message("prompts.requirements_evaluation.system_instruction", language),
        },
        {"role": "user", "content": user_content},
    ]
    is_sufficient = False
    if state.get("hearing_count", 1) >= MAX_HEARING_COUNT:
        is_sufficient = True
        print("[requirements_evaluation] Reached MAX_HEARING_COUNT, forcing to sufficient")
    else:
        resp = llm_chat.invoke(messages)
        ans = _to_text(resp.content).strip().lower()
        is_sufficient = "yes" in ans
        print("[requirements_evaluation] answer:", ans, "=> is_sufficient:", is_sufficient)

    # ヒアリングの必要がない場合は、要件サマリに進む
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

    # ヒアリングを続ける場合は、簡単な質問を1問だけ投げる
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
    print("[hearing] messages to LLM:", messages)
    resp = await llm_chat.ainvoke(messages)
    print("[hearing] response from LLM:", resp.content)
    question = _to_text(resp.content).strip() or get_message("prompts.hearing.fallback_question", language)

    return {
        "messages": [{"role": "assistant", "content": question}],
        "hearing_count": state.get("hearing_count", 1) + 1,
        "phase": Phase.HEARING.value,
    }


def should_hear_again(state: State) -> str:
    """ヒアリング継続判定: 'yes' or 'no' を返す（同期関数）"""
    phase = state.get("phase", "")
    if phase == Phase.HEARING.value:
        return "yes"
    return "no"


async def summarizing(state: State):
    """要件サマリ生成 ノード

    ヒアリング会話全体から要件サマリを生成し state に格納する。"""
    language = state.get("language", DEFAULT_LANGUAGE)

    # 会話履歴を文字列化（長すぎる場合は適度にトリム）
    MAX_DIALOG_HISTORY = 8000
    dialog_history = _make_dialog_history(state)
    if len(dialog_history) > MAX_DIALOG_HISTORY:
        dialog_history = dialog_history[-MAX_DIALOG_HISTORY:]

    # 要件サマリ生成プロンプト
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

    # LLM 呼び出し
    print("[summarizing] messages to LLM:", messages)
    resp = await llm_chat.ainvoke(messages)
    summary_text = _to_text(resp.content).strip()
    print("[summarizing] response from LLM:", summary_text)

    # 要件サマリの作成に失敗した場合は会話履歴をそのまま使う
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
        "phase": Phase.CODE_GENERATING.value,  # 次は code_generating に進む
    }


def _retrieve_code_generation_context(lint_messages: List[BicepLintMessage], bicep_code: str) -> str:
    """Lint メッセージに基づいて修正ポイントを記載したコンテキスト文字を取得する"""
    MAX_BICEP_CODE = 8000
    MAX_LINT_MESSAGES_FOR_CONTEXT = 3

    bicep_code_for_context = (
        bicep_code if len(bicep_code) <= MAX_BICEP_CODE else bicep_code[:MAX_BICEP_CODE] + "\n(... truncated)"
    )

    # lint も code も空なら、context は必要ない
    if not lint_messages and not bicep_code:
        return ""
    # lint のみが空なら、code を context として返す
    if bicep_code and not lint_messages:
        return f"## Bicep Code\n{bicep_code_for_context}"
    # code のみが空なら、lint メッセージを返す (通常はありえないはず)
    if lint_messages and not bicep_code:
        print("[_retrieve_code_generation_context] Lint messages but no Bicep code")
        return "## Lint Messages\n" + "\n".join([str(m) for m in lint_messages])

    context_lines = []
    bicep_code_blocks = parse_bicep_blocks(bicep_code)

    for lint_message in lint_messages[:MAX_LINT_MESSAGES_FOR_CONTEXT]:
        print(f"[_retrieve_code_generation_context] Processing lint message: {lint_message}")

        line = lint_message.line

        # 該当行を含むブロックを検索する
        block = None
        for b in bicep_code_blocks:
            if b.start_line <= line <= b.end_line:
                block = b
                break

        # 該当するコードブロックが見つからない場合
        if not block:
            print(f"[_retrieve_code_generation_context] No code block found for lint message: {lint_message}")
            context_lines.append(f"## Lint Message at line {line}\n{str(lint_message)}")
            continue

        # リソース ブロック以外の場合
        if not block.kind == "resource":
            print(f"[_retrieve_code_generation_context] Block at line {line} is not a resource, kind={block.kind}")
            context_lines.append(f"## Lint Message at line {line}\n{str(lint_message)}")
            continue

        # リソース ブロックの場合
        # リソースプロバイダ、タイプ、API バージョンを抽出
        resource_name = ""
        resource_type = ""
        api_version = ""
        m = re.match(r"resource\s+(\w+)\s+'([^@]+)@([^']+)'", block.text)
        if m:
            resource_name = m.group(1)
            resource_type = m.group(2)
            api_version = m.group(3)
            print(
                f"[_retrieve_code_generation_context] Found resource block for lint message at line {line}: {resource_name}, {resource_type}, {api_version}"
            )

        # リソース文字列がパースできなかった場合
        if not resource_type:
            print(f"[_retrieve_code_generation_context] Failed to parse resource block for lint message at line {line}")
            context_lines.append(f"## Lint Message at line {line}\n{str(lint_message)}")
            continue

        # 公式ドキュメントの該当ページを取得
        # Sample: https://learn.microsoft.com/en-us/azure/templates/Microsoft.Network/virtualNetworks/virtualNetworkPeerings?pivots=deployment-language-bicep
        docs_url = f"https://learn.microsoft.com/en-us/azure/templates/{resource_type.lower()}/?pivots=deployment-language-bicep"
        docs_content = fetch_url_content(
            docs_url,
            timeout=5.0,
            max_bytes=1_000_000,
            query_selector='div[data-pivot="deployment-language-bicep"]',
            compressed=True,
        )
        print(f"[_retrieve_code_generation_context] Fetched docs content from {docs_url}, {len(docs_content)} bytes")

        # Lint メッセージ、コードブロック、ドキュメント を基に、改善案を LLM に問い合わせる
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
        print("[_retrieve_code_generation_context] Calling LLM for guidance")
        resp = llm_code.invoke(messages)
        guidance = _to_text(resp.content).strip()
        print(f"[_retrieve_code_generation_context] Guidance from LLM:\n{guidance}")
        if not guidance:
            context_lines.append(f"## Lint Message at line {line}\n{str(lint_message)}")
            continue
        context_lines.append(
            f"## Lint Message at line {line}\n### Lint Message\n{str(lint_message)}\n### Guidance\n{guidance}"
        )

    print("[_retrieve_code_generation_context] context_lines:", context_lines)
    return f"## Bicep Code\n{bicep_code_for_context}" + "\n\n" + "\n\n".join(context_lines)


async def code_generating(state: State):
    """Bicep コード生成 / 再生成 ノード

    - 要件サマリと直近の lint 結果を踏まえてコード生成
    - 直近生成コードがあれば再生成プロンプトを利用
    """
    language = state.get("language", DEFAULT_LANGUAGE)
    requirement_summary = state.get("requirement_summary")
    assert (
        isinstance(requirement_summary, str) and requirement_summary.strip()
    ), "requirement_summary はコード生成に必須"
    lint_messages = state.get("lint_messages", [])
    bicep_code = state.get("bicep_code", "")
    generation_count = state.get("generation_count", 0)

    # プロンプト構築
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

    # その他のコンテキストを付与
    context = _retrieve_code_generation_context(lint_messages, bicep_code)
    user_prompt += context
    print("[code_generating] context:", context)

    # LLM 呼び出し
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    print("[code_generating] calling LLM to generate a bicep code")
    resp = await llm_code.ainvoke(messages)
    print("[code_generating] got response from LLM")

    # コードブロックがあれば中身だけ抽出
    code_text = _to_text(resp.content).strip()
    m = re.search(r"```(?:bicep)?\s*\n([\s\S]*?)```", code_text, flags=re.IGNORECASE)
    if m:
        code_text = m.group(1).strip()
    print("[code_generating] generated code:", code_text)

    return {
        "messages": [{"role": "assistant", "content": chat_message}],
        "bicep_code": code_text,
        "generation_count": generation_count + 1,
        "phase": Phase.CODE_VALIDATING.value,  # 次は code_validating に進む
    }


async def code_validating(state: State):
    """Bicep コード検証 ノード

    生成されたコードを一時ファイルに保存し、`az bicep lint` コマンドで検証する。
    検証結果をパースして state に格納する。
    """
    language = state.get("language", DEFAULT_LANGUAGE)
    chat_message = ""

    def _get_bicep_command() -> str:
        if shutil.which("az") is not None:
            return "az bicep"
        return "bicep"

    # 生成コードの取得
    # 編集画面でコードが空になっている場合もあるので、その場合は検証せずに終了する
    code = state.get("bicep_code", "")
    if not code:
        return {
            "messages": [{"role": "assistant", "content": get_message("messages.code_validation.no_code", language)}],
            "lint_messages": [],
            "validation_passed": False,
        }

    # 一時ファイルに保存して lint 実行
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

            # lint 結果のパース
            lint_messages = parse_bicep_lint_output(lint_output_raw)
            print("[code_validating] lint_messages:", lint_messages)

            # 検証合格かどうか
            validation_passed = len(lint_messages) == 0

            # プレビューは最大10件に制限
            MAX_LINT_MESSAGES_FOR_PREVIEW = 10
            preview_lines = [str(lint_msg) for lint_msg in lint_messages[:MAX_LINT_MESSAGES_FOR_PREVIEW]]
            if len(lint_messages) > MAX_LINT_MESSAGES_FOR_PREVIEW:
                preview_lines.append(f"... ({len(lint_messages)-MAX_LINT_MESSAGES_FOR_PREVIEW} more)")
            lint_output = "\n".join(preview_lines)

            # ユーザーメッセージの構築
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

    # 次のフェーズ決定
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
    """完了 ノード

    - ヒアリング or コード生成のどちらかで完了した場合に到達する
    - 再生成上限に達した場合もここに来る
    - 最終メッセージを返すだけ
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
# グラフ構築（checkpointer 付き：SQLite → 無ければ Memory に自動フォールバック）
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
            print("[graph] Using SqliteSaver(checkpoints.db)")
    elif MemorySaver is not None:
        checkpointer = MemorySaver()
        if DEBUG_LOG:
            print("[graph] Using MemorySaver (no persistence)")
    else:
        raise RuntimeError("No available checkpointer: SqliteSaver and MemorySaver are both unavailable")

    return gb.compile(checkpointer=checkpointer)


GRAPH = build_graph()


# ──────────────────────────────────────────────────────────────────────────────
# ルーティング
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
    会話リセット（簡易版）
    - 実運用はフロントで毎会話UUIDを割り当て、thread_idを変更するのが最も安全
    - ここでは messages, bicep_code などを空にする
    """
    config = {"configurable": {"thread_id": session_id or "default"}}  # type: ignore[assignment]
    GRAPH.update_state(  # type: ignore[arg-type]
        config,  # type: ignore[arg-type]
        initial_state(),
    )
    return {"message": f"会話({session_id})がリセットされました"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    使い方:
      1) POST /chat {"session_id":"abc","message":"ストレージがほしい"}
         → 質問が返る（is_complete=false）
      2) POST /chat {"session_id":"abc","message":"Webアプリ用です"}
         → 次の質問（is_complete=false）
      3) …繰り返し
      4) 十分な要件になると code_generation が走り、bicep_code + is_complete=true を返す
    """
    try:
        session_id = request.session_id or "default"
        config = {"configurable": {"thread_id": session_id}}  # type: ignore[assignment]

        print(f"[chat_endpoint] session_id={session_id} request.message={request.message!r}")

        # ① 言語設定とHumanの発話を state に追加
        language = request.language or DEFAULT_LANGUAGE
        state_update = {"language": language}
        if request.message:
            state_update["messages"] = [{"role": "user", "content": request.message}]  # type: ignore[assignment]

        # セッション初期化（stateが空の場合のみ）
        ensure_session_initialized(config, language)

        GRAPH.update_state(  # type: ignore[arg-type]
            config,  # type: ignore[arg-type]
            state_update,
        )

        # ② グラフを進める（非同期ストリームの最初の1つだけ取得して即抜ける）
        state = GRAPH.get_state(config).values  # type: ignore[arg-type]
        async for _chunk in GRAPH.astream(None, config=config, stream_mode="updates"):  # type: ignore[arg-type]
            break

        # ③ 現在stateを取得
        state = GRAPH.get_state(config).values  # type: ignore[arg-type]
        messages = state.get("messages", [])
        bicep_code = state.get("bicep_code") or ""
        phase = state.get("phase", "")
        requirement_summary = state.get("requirement_summary", "")

        # 直近のAI発話
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

        # requires_user_input 判定
        # - True: ユーザーの回答待ちが必要なとき (主に hearing の質問)
        # - False: 自動で次ステップへ進めたい中間状態
        requires_user_input = phase in {p.value for p in REQUIRE_USER_INPUT_PHASES}
        is_completed = phase == Phase.COMPLETED.value
        if is_completed:
            requires_user_input = False

        # 完了時メッセージのデフォルト
        default_msg = "Bicepコードの生成が完了しました！" if is_completed else "次の質問を用意しています…"

        return ChatResponse(
            message=latest_ai_text or default_msg,
            bicep_code=bicep_code,
            phase=phase,
            requirement_summary=requirement_summary,
            requires_user_input=requires_user_input,
        )

    except Exception as e:  # noqa
        if DEBUG_LOG:
            print("[chat] ERROR:", repr(e))
        raise HTTPException(status_code=500, detail=f"エラーが発生しました: {str(e)}")


# ──────────────────────────────────────────────────────────────────────────────
# dev run
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    # Windows環境では明示的にlocalhostを指定すると楽
    uvicorn.run(app, host="127.0.0.1", port=8000)
