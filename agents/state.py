from typing import Annotated, Any, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class TravelState(TypedDict, total=False):
    # 通过 add_messages 在节点间自动追加消息
    messages: Annotated[Sequence[BaseMessage], add_messages]
    intent: str
    city: str
    days: int
    preference: str
    raw_materials: str
    missing_fields: list[str]
    router_reason: str
    router_model: str
    planner_model: str
    user_query: str
    vector_db: Any
    