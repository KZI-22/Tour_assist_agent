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

    if intent == "need_answer":
        if not raw_materials:
            return {
                "messages": [
                    AIMessage(content="资料不足，暂时无法准确回答这个问题。")
                ]
            }

        planner_model = (state.get("planner_model") or "glm-4.5-air").strip()
        llm = get_llm(planner_model)
        prompt = f"""
你是旅游问答助手。请基于提供资料，直接回答用户当前问题，不要强行输出行程模板。

【用户问题】
{user_query or '请根据资料给出结论'}

【资料】
{raw_materials}

【回答要求】
1. 先给结论，再补充依据。
2. 资料不足时明确说明，不编造。
3. 使用简洁 Markdown。
"""
        try:
            response = llm.invoke(prompt)
            content = getattr(response, "content", None) or str(response)
            return {"messages": [AIMessage(content=content)]}
        except Exception as exc:
            return {
                "messages": [
                    AIMessage(content=f"问答生成失败，请稍后重试。错误信息：{exc}")
                ]
            }

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
3. 每个时间段给出：地点/活动、推荐理由、建议停留时长、交通建议。
4. 每天补充 1-2 个餐饮建议。
5. 最后给出“注意事项”与“预算建议（低/中/高三档）”。
6. 不要编造资料中完全不存在的硬性事实；不确定信息用“建议/可考虑”表述。
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