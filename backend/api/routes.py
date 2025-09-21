from fastapi import APIRouter, HTTPException
from backend.core.state import ChatRequest, ChatResponse, initial_state
from backend.domain import Phase, REQUIRE_USER_INPUT_PHASES, DEFAULT_LANGUAGE
from backend.core.logging import logger
from backend.graph.graph import GRAPH, ensure_session_initialized

router = APIRouter()


@router.get("/")
async def root():
    return {"message": "Bicep Generator API is running"}


@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "bicep-generator-api"}


@router.get("/config")
async def get_config():
    from backend.core.config import (
        AZURE_OPENAI_ENDPOINT,
        CHAT_DEPLOYMENT_NAME,
        CODE_DEPLOYMENT_NAME,
        OPENAI_API_VERSION,
    )

    return {
        "azure_openai_endpoint": AZURE_OPENAI_ENDPOINT,
        "chat_deployment_name": CHAT_DEPLOYMENT_NAME,
        "code_deployment_name": CODE_DEPLOYMENT_NAME,
        "openai_api_version": OPENAI_API_VERSION,
    }


@router.post("/reset")
async def reset_conversation(session_id: str = "default"):
    config = {"configurable": {"thread_id": session_id or "default"}}  # type: ignore[assignment]
    GRAPH.update_state(config, initial_state())  # type: ignore[arg-type]
    return {"message": f"The conversation (session_id: {session_id}) has been reset."}


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        session_id = request.session_id or "default"
        config = {"configurable": {"thread_id": session_id}}  # type: ignore[assignment]
        logger.debug("session_id=%s request.message=%r", session_id, request.message)
        language = request.language or DEFAULT_LANGUAGE
        state_update = {"language": language}
        if request.message:
            state_update["messages"] = [{"role": "user", "content": request.message}]  # type: ignore[assignment]
        ensure_session_initialized(config, language)
        GRAPH.update_state(config, state_update)  # type: ignore[arg-type]
        async for _chunk in GRAPH.astream(None, config=config, stream_mode="updates"):  # type: ignore[arg-type]
            break
        state = GRAPH.get_state(config).values  # type: ignore[arg-type]
        messages = state.get("messages", [])
        bicep_code = state.get("bicep_code") or ""
        phase = state.get("phase", "")
        requirement_summary = state.get("requirement_summary", "")
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
        requires_user_input = phase in {p.value for p in REQUIRE_USER_INPUT_PHASES}
        is_completed = phase == Phase.COMPLETED.value
        if is_completed:
            requires_user_input = False
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
