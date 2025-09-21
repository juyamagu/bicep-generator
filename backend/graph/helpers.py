from typing import Any, List

# Utility functions extracted from original app.py


def to_text(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: List[str] = []
        for r in raw:
            if isinstance(r, str):
                parts.append(r)
            elif isinstance(r, dict):
                val = r.get("text") or r.get("content") or ""
                if isinstance(val, list):
                    parts.append(" ".join(str(v) for v in val))
                else:
                    parts.append(str(val))
            else:
                parts.append(str(r))
        return "\n".join(p for p in parts if p)
    return str(raw)


def make_dialog_history(state) -> str:
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


__all__ = ["to_text", "make_dialog_history"]
