from langgraph.graph import StateGraph, END
from agents.state import TravelState
from agents.router_node import router_agent
from agents.research_node import researcher_agent
from agents.planner_node import planner_agent


def _route_after_router(state: TravelState) -> str:
    """根据 router 的 intent 选择后续路径。"""
    intent = (state.get("intent") or "").strip().lower()

    if intent in {"need_plan", "need_answer"}:
        return "researcher"

    # need_more_info / general_chat / other 由 router 直接回复并结束
    return END

def build_travel_graph():
    """构建旅行多节点工作流：router -> researcher -> planner。"""
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

    workflow.add_edge("researcher", "planner")
    workflow.add_edge("planner", END)

    return workflow.compile()