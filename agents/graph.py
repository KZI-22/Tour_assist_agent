from langgraph.graph import END, StateGraph

from agents.planner_node import planner_agent
from agents.research_node import researcher_agent
from agents.router_node import router_agent
from agents.state import TravelState


def _route_after_router(state: TravelState) -> str:
    """Route requests after intent parsing."""
    intent = (state.get("intent") or "").strip().lower()

    # need_plan 和 need_answer 都需要经过 researcher 收集信息
    if intent in {"need_plan", "need_answer"}:
        return "researcher"

    # 其他意图（general_chat, other, need_more_info）直接结束
    return END


def _route_after_researcher(state: TravelState) -> str:
    """Skip planner for direct answers and continue for trip planning."""
    intent = (state.get("intent") or "").strip().lower()

    if intent == "need_plan":
        return "planner"

    return END


def build_travel_graph():
    """Build the travel workflow graph."""
    workflow = StateGraph(TravelState)

    workflow.add_node("router", router_agent)
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("planner", planner_agent)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "researcher": "researcher",
            END: END,
        },
    )

    workflow.add_conditional_edges(
        "researcher",
        _route_after_researcher,
        {
            "planner": "planner",
            END: END,
        },
    )
    workflow.add_edge("planner", END)

    return workflow.compile()
