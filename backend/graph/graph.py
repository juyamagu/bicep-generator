from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages, AnyMessage  # noqa: F401
from langchain_core.messages import AIMessage  # noqa: F401
from backend.domain import Phase, DEFAULT_LANGUAGE
from backend.core.state import State, initial_state
from backend.core.config import DEBUG_LOG
from backend.core.logging import logger
from backend.core.llm import llm_chat, llm_code  # noqa: F401  (keep to ensure early init / failures surface)
from backend.graph.nodes import hearing, should_hear_again, summarizing, code_generation, code_validation, completed

SqliteSaver = None
MemorySaver = None
try:
    from langgraph.checkpoint.sqlite import SqliteSaver as _SqliteSaver  # type: ignore

    SqliteSaver = _SqliteSaver
except Exception:  # pragma: no cover
    try:
        from langgraph.checkpoint.memory import MemorySaver as _MemorySaver  # type: ignore

        MemorySaver = _MemorySaver
    except Exception:  # pragma: no cover
        pass


def build_graph():
    gb = StateGraph(State)

    # Nodes
    gb.add_node(Phase.HEARING.value, hearing)
    gb.add_node(Phase.SUMMARIZING.value, summarizing)
    gb.add_node(Phase.CODE_GENERATING.value, code_generation)
    gb.add_node(Phase.CODE_VALIDATING.value, code_validation)
    gb.add_node(Phase.COMPLETED.value, completed)

    # Edges
    gb.set_entry_point(Phase.HEARING.value)
    gb.add_conditional_edges(
        Phase.HEARING.value,
        should_hear_again,
        {"yes": Phase.HEARING.value, "no": Phase.SUMMARIZING.value},
    )
    gb.add_edge(Phase.SUMMARIZING.value, Phase.CODE_GENERATING.value)
    gb.add_edge(Phase.CODE_GENERATING.value, Phase.CODE_VALIDATING.value)
    gb.add_conditional_edges(
        Phase.CODE_VALIDATING.value,
        lambda s: "yes" if s.get("phase") == Phase.CODE_GENERATING.value else "no",
        {"yes": Phase.CODE_GENERATING.value, "no": Phase.COMPLETED.value},
    )
    gb.set_finish_point(Phase.COMPLETED.value)

    # States that require user input to proceed
    if SqliteSaver is not None:
        checkpointer = SqliteSaver("checkpoints.db")
        if DEBUG_LOG:
            logger.debug("Using SqliteSaver(checkpoints.db)")
    elif MemorySaver is not None:
        checkpointer = MemorySaver()
        if DEBUG_LOG:
            logger.debug("Using MemorySaver (no persistence)")
    else:
        raise RuntimeError("No available checkpointer")

    return gb.compile(checkpointer=checkpointer)


GRAPH = build_graph()


def ensure_session_initialized(config, language: str = DEFAULT_LANGUAGE):
    try:
        existing = GRAPH.get_state(config).values  # type: ignore[arg-type]
        if not existing or not existing.get("phase"):
            GRAPH.update_state(config, initial_state(language))  # type: ignore[arg-type]
    except Exception:
        try:
            GRAPH.update_state(config, initial_state(language))  # type: ignore[arg-type]
        except Exception:
            pass


__all__ = ["GRAPH", "build_graph", "ensure_session_initialized"]
