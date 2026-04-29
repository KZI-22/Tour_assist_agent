from langchain_core.messages import AIMessage

from agents.state import TravelState
from core.llm_core import get_llm


def planner_agent(state: TravelState) -> dict:
    """根据 researcher 汇总素材，生成最终行程文本。"""
    intent = (state.get("intent") or "").strip().lower()
    user_query = (state.get("user_query") or "").strip()
    city = (state.get("city") or "").strip()
    days = int(state.get("days") or 0)
    preference = (state.get("preference") or "综合").strip()
    raw_materials = (state.get("raw_materials") or "").strip()

    # 注意：need_answer 意图在 researcher 后直接结束，不会经过 planner
    # 所以这里只处理 need_plan 意图

    if not city:
        return {
            "messages": [
                AIMessage(content="还缺少目的地城市，暂时无法生成完整行程。")
            ]
        }

    if not raw_materials:
        return {
            "messages": [
                AIMessage(content="暂未采集到有效资料，请稍后重试或补充更具体需求。")
            ]
        }

    planner_model = (state.get("planner_model") or "glm-4.5-air").strip()
    llm = get_llm(planner_model)

    prompt = f"""
你是一位资深旅游规划师。请基于提供的资料，生成可执行的 {city} 行程方案。

【用户约束】
- 目的地：{city}
- 天数：{days if days > 0 else '未指定'}
- 偏好：{preference}

【已采集资料】
{raw_materials}

【输出要求】
1. 使用 Markdown。
2. 按天拆分；每一天包含上午、下午、晚上。
3. **每天开头必须标注当天的天气预报信息**（天气状况、温度范围），格式如：「🌤 天气：晴转多云，15-23℃」。
4. **天气适配规则（重要）**：
   - 如果某天预报有雨、雪、雷暴、冰雹、大雾、霾、沙尘等恶劣天气，该天的上午/下午时段**必须安排室内活动**（如博物馆、室内景点、商场、美食探店、文化体验等），避免安排户外徒步、公园游览、户外拍照等。
   - 恶劣天气的晚上可以安排室内餐饮或演出。
   - 仅在天气良好时才推荐户外景点和活动。
5. 每个时间段给出：地点/活动、推荐理由、建议停留时长、交通建议。
6. 每天补充 1-2 个餐饮建议。
7. 最后给出”注意事项”与”预算建议（低/中/高三档）”。
8. 不要编造资料中完全不存在的硬性事实；不确定信息用”建议/可考虑”表述。
"""

    try:
        response = llm.invoke(prompt)
        content = getattr(response, "content", None) or str(response)
        return {"messages": [AIMessage(content=content)]}
    except Exception as exc:
        return {
            "messages": [
                AIMessage(content=f"行程生成失败，请稍后重试。错误信息：{exc}")
            ]
        }