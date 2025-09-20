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

try:
    from .constants import Phase, AUTO_PROGRESS_PHASES, DEFAULT_LANGUAGE  # type: ignore
    from .i18n.messages import get_message  # type: ignore
except ImportError:  # 実行方法によっては相対 import が失敗する場合に備える
    from constants import Phase, AUTO_PROGRESS_PHASES, DEFAULT_LANGUAGE  # type: ignore
    from i18n.messages import get_message  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
# Checkpointer: SqliteSaver が無い環境では自動で MemorySaver に切替
# ──────────────────────────────────────────────────────────────────────────────
SqliteSaver = None
MemorySaver = None
try:
    from langgraph.checkpoint.sqlite import SqliteSaver as _SqliteSaver

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
# AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
CHAT_DEPLOYMENT_NAME = os.getenv("CHAT_DEPLOYMENT_NAME", "gpt-4.1")
CODE_DEPLOYMENT_NAME = os.getenv("CODE_DEPLOYMENT_NAME", "gpt-4.1")

DEBUG_LOG = os.getenv("DEBUG_LOG", "0") in ("1", "true", "True")
DEBUG_LOG = True
MAX_HEARING_CALLS = int(os.getenv("MAX_HEARING_CALLS", "20"))  # ヒアリングの最大回数
MAX_REGEN_CALLS = int(os.getenv("MAX_REGEN_CALLS", "5"))  # コード再生成（lint後）の最大回数

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
    n_callings: int
    current_user_message: str
    bicep_code: str  # 最新生成コード
    lint_output: str  # lint 結果
    validation_passed: bool  # lint 合格
    code_regen_count: int  # 再生成回数（初回生成は含めない）
    phase: str  # hearing | code_generation | code_validation | completed
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
    # フロントエンドが次のユーザー入力を待つ必要があるか（True=待つ）
    # hearing フェーズの質問などユーザー回答が必要な場面で True
    # 自動で次ステップに進めたい中間メッセージ（要件サマリ表示、コード生成完了、lint 結果表示 等）は False
    requires_user_input: bool = True


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


def _history_from_state(state: State) -> List[Dict[str, str]]:
    history: List[Dict[str, str]] = []
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
            history.append({"role": role, "content": _to_text(content_val)})
    return history


async def hearing(state: State):
    """要件ヒアリング: 1問だけ短く投げる"""
    language = state.get("language", DEFAULT_LANGUAGE)
    history = _history_from_state(state)
    messages = [
        {
            "role": "system",
            "content": get_message("prompts.hearing.system_role", language),
        },
        *history,
        {
            "role": "user",
            "content": get_message("prompts.hearing.user_instruction", language),
        },
    ]
    resp = await llm_chat.ainvoke(messages)
    question = _to_text(resp.content).strip() or get_message("prompts.hearing.fallback_question", language)
    return {
        "messages": [{"role": "assistant", "content": question}],
        "n_callings": state.get("n_callings", 0) + 1,
        "current_user_message": state.get("current_user_message", ""),
        "bicep_code": state.get("bicep_code", ""),
        "lint_output": state.get("lint_output", ""),
        "validation_passed": state.get("validation_passed", False),
        "phase": Phase.HEARING.value,
        "requirement_summary": state.get("requirement_summary", ""),
        "language": language,
    }


def should_hear_again(state: State) -> str:
    """ヒアリング継続判定: 'yes' or 'no' を返す（同期関数）"""
    language = state.get("language", DEFAULT_LANGUAGE)
    history = _history_from_state(state)
    messages = [
        {
            "role": "system",
            "content": get_message("prompts.should_hear_again.system_instruction", language),
        },
        *history,
        {"role": "user", "content": get_message("prompts.should_hear_again.user_question", language)},
    ]
    if state.get("n_callings", 0) >= MAX_HEARING_CALLS:
        return "no"
    resp = llm_chat.invoke(messages)
    ans = _to_text(resp.content).strip().lower()
    return "yes" if "yes" in ans else "no"


async def code_generation(state: State):
    """Bicep コード生成 (lint 結果があれば考慮)"""
    language = state.get("language", DEFAULT_LANGUAGE)
    requirement_summary = state.get("requirement_summary") or "(要件サマリ未生成)"
    lint_output_full = state.get("lint_output") or "(lint 結果なし)"
    previous_code = state.get("bicep_code", "")
    prev_code_exists = bool(previous_code)
    regen_count = state.get("code_regen_count", 0) + (1 if prev_code_exists else 0)

    # 大きすぎる入力はトリム（LLM のトークン節約）
    lint_output = lint_output_full[:3000]
    truncated_note = ""
    if len(lint_output_full) > len(lint_output):
        truncated_note = "\n(※ lint 出力は長いため一部のみ使用)"

    prev_code_section = ""
    if prev_code_exists:
        # 旧コードも長過ぎればトリム
        prev_code_for_prompt = previous_code
        max_prev_len = 8000
        if len(prev_code_for_prompt) > max_prev_len:
            prev_code_for_prompt = prev_code_for_prompt[-max_prev_len:]
            prev_code_section = (
                "## 直近生成コード (末尾" + str(max_prev_len) + "文字)\n```bicep\n" + prev_code_for_prompt + "\n```\n"
            )
        else:
            prev_code_section = f"## 直近生成コード\n```bicep\n{prev_code_for_prompt}\n```\n"

    # 初回と再生成で system プロンプトを少し分岐
    if not prev_code_exists:
        system_prompt = get_message("prompts.code_generation.system_initial", language)
    else:
        system_prompt = get_message("prompts.code_generation.system_regeneration", language)

    user_prompt_parts = [
        f"## 要件サマリ\n{requirement_summary}",
    ]
    if prev_code_section:
        user_prompt_parts.append(prev_code_section)
    user_prompt_parts.append(f"## 直近 lint 結果\n{lint_output}{truncated_note}")
    if prev_code_exists:
        user_prompt_parts.append(get_message("prompts.code_generation.user_regeneration", language))
    else:
        user_prompt_parts.append(get_message("prompts.code_generation.user_initial", language))

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(user_prompt_parts)},
    ]

    if DEBUG_LOG:
        print("[code generation prompt]", messages)

    resp = await llm_code.ainvoke(messages)
    text = _to_text(resp.content).strip()
    code_text = text
    m = re.search(r"```(?:bicep)?\s*\n([\s\S]*?)```", text, flags=re.IGNORECASE)
    if m:
        code_text = m.group(1).strip()
    done_msg = (
        get_message("messages.code_generation.initial_done", language)
        if not prev_code_exists
        else get_message("messages.code_generation.regeneration_done", language)
    )
    return {
        "messages": [{"role": "assistant", "content": done_msg}],
        "n_callings": state.get("n_callings", 0),
        "current_user_message": state.get("current_user_message", ""),
        "bicep_code": code_text,
        "lint_output": state.get("lint_output", ""),
        "validation_passed": False,
        "code_regen_count": regen_count,
        "phase": Phase.CODE_GENERATION.value,
        "requirement_summary": requirement_summary,
        "language": language,
    }


async def summarize_requirements(state: State):
    """ヒアリング会話全体から要件サマリを生成し state に格納する。"""
    language = state.get("language", DEFAULT_LANGUAGE)
    history = _history_from_state(state)
    # 会話履歴を文字列化（長すぎる場合は適度にトリム）
    joined = []
    for h in history:
        role = h.get("role")
        content = h.get("content", "")
        if role == "user":
            joined.append(f"[USER] {content}")
        elif role == "assistant":
            joined.append(f"[ASSISTANT] {content}")
    raw_dialogue = "\n".join(joined)
    if len(raw_dialogue) > 8000:
        raw_dialogue = raw_dialogue[-8000:]

    messages = [
        {
            "role": "system",
            "content": get_message("prompts.summarize_requirements.system_instruction", language),
        },
        {
            "role": "user",
            "content": get_message(
                "prompts.summarize_requirements.user_instruction", language, raw_dialogue=raw_dialogue
            ),
        },
    ]
    resp = await llm_chat.ainvoke(messages)
    summary_text = _to_text(resp.content).strip()
    if not summary_text:
        summary_text = get_message("messages.summarize_requirements.generation_failed", language)

    if DEBUG_LOG:
        print("[requirement summary]", summary_text)

    display_msg = get_message("messages.summarize_requirements.display_message", language, summary_text=summary_text)
    return {
        "messages": [{"role": "assistant", "content": display_msg}],
        "requirement_summary": summary_text,
        "phase": Phase.SUMMARIZING.value,
        "language": language,
    }


async def code_validation(state: State):

    def get_bicep_command() -> str:
        if shutil.which("az") is not None:
            return "az bicep"
        return "bicep"

    code = state.get("bicep_code", "")
    if not code:
        return {
            "messages": [{"role": "assistant", "content": "検証対象コードがありません。"}],
            "lint_output": "(no code)",
            "validation_passed": False,
        }

    tmp_path = None
    lint_output = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bicep", mode="w", encoding="utf-8") as f:
            f.write(code)
            tmp_path = f.name
        try:
            bicep_cmd = get_bicep_command()
            proc = subprocess.run(
                " ".join([bicep_cmd, "lint", "--file", tmp_path]),
                capture_output=True,
                text=True,
                timeout=60,
                shell=True,
            )
            lint_output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        except FileNotFoundError:
            lint_output = "bicep コマンドが見つかりません。Azure CLI や Bicep 拡張のインストールを確認してください。"
        except subprocess.TimeoutExpired:
            lint_output = "az bicep lint がタイムアウトしました。"
        except Exception as e:  # noqa
            lint_output = f"lint 実行エラー: {e}"
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    if DEBUG_LOG:
        print("[code_validation] lint_output:", lint_output)
    lint_output_preview = lint_output.strip()
    if len(lint_output_preview) > 1200:
        lint_output_preview = lint_output_preview[:1200] + "... (truncated)"

    # バリデーションの判定
    # LLM に判定させるやり方も試したが、単純に error/failed の有無で判定するので問題ないと判断した
    #
    # validation_passed = False
    # messages = [
    #     {
    #         "role": "system",
    #         "content": "あなたは Bicep コードの品質ゲート判定者です。Lint 結果を見て、致命的または重要な問題があって再度コード修正の必要がある場合は 'failed'、無ければ 'passed' で答えてください。回答は 'failed' または 'passed' のみ。",
    #     },
    #     {
    #         "role": "user",
    #         "content": f"""## Lint Output\n{lint_output}\n""",
    #     },
    # ]
    # try:
    #     resp = llm.invoke(messages)
    #     answer = _to_text(resp.content).strip().lower()
    #     if "passed" in answer:
    #         validation_passed = True
    # except Exception:
    #     pass
    validation_passed = not bool(re.search(r"\b(error|warning)\b", lint_output, flags=re.IGNORECASE))

    message = f"lint 結果:\n==========\n{lint_output_preview}\n"
    return {
        "messages": [{"role": "assistant", "content": message}],
        "lint_output": lint_output,
        "validation_passed": validation_passed,
        "phase": Phase.CODE_VALIDATION.value,
    }


def should_regenerate_code(state: State) -> str:
    code = state.get("bicep_code", "")
    if not code:
        return "yes"

    if state.get("code_regen_count", 0) >= MAX_REGEN_CALLS:
        return "no"

    return "yes" if not state.get("validation_passed", False) else "no"


async def finalize_validation(state: State):
    regen_count = state.get("code_regen_count", 0)
    validation_passed = state.get("validation_passed", False)

    def get_message():
        if validation_passed:
            return "Bicep コードは検証を通過しました。出力されたコードをご利用ください！"
        if regen_count >= MAX_REGEN_CALLS:
            return (
                "自動再生成の上限 (MAX_REGEN_CALLS) に達したため処理を終了しました。"
                " 残存する警告/エラーがある場合は手動で修正してください。"
            )
        return "検証をパスしていませんが、処理を終了します。適宜、手動で修正してご利用ください。"

    return {
        "messages": [{"role": "assistant", "content": get_message()}],
        "phase": Phase.COMPLETED.value,
    }


# ──────────────────────────────────────────────────────────────────────────────
# グラフ構築（checkpointer 付き：SQLite → 無ければ Memory に自動フォールバック）
# ──────────────────────────────────────────────────────────────────────────────
def build_graph():
    gb = StateGraph(State)
    gb.add_node("hearing", hearing)
    gb.add_node("summarize_requirements", summarize_requirements)
    gb.add_node("code_generation", code_generation)
    gb.add_node("code_validation", code_validation)
    gb.add_node("finalize_validation", finalize_validation)

    gb.set_entry_point(Phase.HEARING.value)
    gb.add_conditional_edges(
        Phase.HEARING.value,
        should_hear_again,
        {"yes": Phase.HEARING.value, "no": "summarize_requirements"},
    )
    gb.add_edge("summarize_requirements", "code_generation")
    gb.add_edge("code_generation", "code_validation")
    gb.add_conditional_edges(
        "code_validation", should_regenerate_code, {"yes": "code_generation", "no": "finalize_validation"}
    )
    gb.set_finish_point("finalize_validation")

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
        {
            "messages": [],
            "bicep_code": "",
            "n_callings": 0,
            "current_user_message": "",
            "lint_output": "",
            "validation_passed": False,
            "code_regen_count": 0,
            "phase": "hearing",
            "requirement_summary": "",
        },
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

        if DEBUG_LOG:
            print(f"[chat] session={session_id} message={request.message!r}")

        # ① 言語設定とHumanの発話を state に追加
        language = request.language or DEFAULT_LANGUAGE
        state_update = {"language": language}
        if request.message:
            state_update["messages"] = [{"role": "user", "content": request.message}]

        GRAPH.update_state(  # type: ignore[arg-type]
            config,  # type: ignore[arg-type]
            state_update,
        )

        # ② 必要に応じて複数ステップ自動実行する
        steps_executed = 0
        while True:
            steps_executed += 1

            # ループ防止のため最大ステップ数を設定
            if steps_executed > 10:
                if DEBUG_LOG:
                    print("[chat] Max steps executed, breaking loop")
                break

            state = GRAPH.get_state(config).values  # type: ignore[arg-type]
            if DEBUG_LOG:
                print(f"[chat] step {steps_executed} phase={state.get('phase')} calls={state.get('n_callings')}")

            async for _chunk in GRAPH.astream(None, config=config, stream_mode="updates"):  # type: ignore[arg-type]
                break
            break

        # ③ 現在stateを取得（上の while ですでに取得済みだが明示的に変数保持）
        state = GRAPH.get_state(config).values  # type: ignore[arg-type]
        msgs = state.get("messages", [])
        bicep = state.get("bicep_code") or ""
        passed = state.get("validation_passed")
        phase = state.get("phase", "")
        requirement_summary = state.get("requirement_summary", "")

        # 直近のAI発話
        latest_ai_text = None
        for m in reversed(msgs):
            if isinstance(m, dict):
                if m.get("role") == "assistant":
                    latest_ai_text = m.get("content")
                    break
            elif hasattr(m, "type") and getattr(m, "type") == "ai":
                latest_ai_text = getattr(m, "content", None)
                break
            elif isinstance(m, str):
                latest_ai_text = m
                break

        # requires_user_input 判定
        # True: ユーザーの回答待ちが必要なとき (主に hearing の質問)
        # False: 自動で次ステップへ進めたい中間状態
        # completed は自動前進させないが、ユーザー入力も不要なので False
        auto_progress_phases = {p.value for p in AUTO_PROGRESS_PHASES}
        requires_user_input = True
        if phase in auto_progress_phases:
            requires_user_input = False
        if phase == Phase.COMPLETED.value:
            requires_user_input = False
        is_complete = bool((phase == Phase.COMPLETED.value) or bool(passed))
        normalized_phase = phase or (
            Phase.COMPLETED.value if is_complete else (Phase.CODE_GENERATION.value if bicep else Phase.HEARING.value)
        )
        # 完了時メッセージのデフォルト
        default_msg = "Bicepコードの生成が完了しました！" if is_complete else "次の質問を用意しています…"
        return ChatResponse(
            message=latest_ai_text or default_msg,
            bicep_code=(
                bicep if (bicep and normalized_phase in ("code_generation", "code_validation", "completed")) else ""
            ),
            phase=normalized_phase,
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
