from __future__ import annotations

import json
import re
from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from agents.state import TravelState
from core.llm_core import get_llm


VALID_INTENTS = frozenset(
    {"need_plan", "need_more_info", "need_answer", "general_chat", "other"}
)

MISSING_FIELD_LABELS: dict[str, str] = {
    "city": "目的地城市",
    "days": "出行天数",
    "preference": "旅行偏好（如美食 / 自然 / 人文 / 亲子 / 摄影 / 休闲）",
}

CLASSIFY_SYSTEM = """你是旅游助手的路由模块。
请根据对话历史返回一个 JSON 对象，字段如下：

{
  "intent": "need_plan | need_more_info | need_answer | general_chat | other",
  "city": "目的地城市名，没有则为空字符串",
  "days": 0,
  "preference": "旅行偏好，多个用 + 连接，没有则为空字符串",
  "reason": "一句话说明判断依据"
}

意图定义：
- need_plan: 用户明确要你生成旅游计划/攻略/行程，并且信息基本齐全
- need_more_info: 用户想规划行程，但目的地 / 天数 / 偏好不完整
- need_answer: 用户在问具体问题，例如天气、景点、美食、交通、当前位置、图片识别、语音内容、知识库内容、个人偏好
- general_chat: 问候、感谢、闲聊
- other: 无法归类

如果用户是在问“我的偏好是什么”“知识库里写了什么”“根据我上传的资料回答”等，这属于 need_answer，不属于 need_plan。

只输出 JSON，不要输出额外说明。"""

PLAN_HINTS = frozenset(
    ("旅游", "旅行", "攻略", "行程", "规划", "安排", "几日游", "自由行", "路线规划")
)
ANSWER_HINTS = frozenset(
    (
        "天气",
        "景点",
        "餐厅",
        "美食",
        "门票",
        "距离",
        "路线",
        "交通",
        "推荐",
        "当前位置",
        "我现在在哪里",
        "我在哪里",
        "识别",
        "偏好",
        "喜好",
        "喜欢什么",
        "知识库",
        "资料",
        "文档",
        "rag",
    )
)
QUESTION_HINTS = frozenset(("什么", "怎么", "如何", "哪里", "哪儿", "几", "多少", "多远", "怎么样", "为何", "吗", "呢"))
CHAT_HINTS = frozenset(("你好", "hi", "hello", "在吗", "谢谢", "再见", "拜拜"))
PROFILE_HINTS = frozenset(("偏好", "喜好", "喜欢", "知识库", "资料", "文档", "rag", "画像"))

MEDIA_RE = re.compile(
    r"(?:用户上传了.*路径[:：][^\n]+\.(?:jpg|jpeg|png|webp|bmp|gif|mp3|wav|m4a|ogg|webm))",
    re.IGNORECASE,
)
CITY_PATTERNS = [
    re.compile(r"(?:去|到|在|想去|前往)([\u4e00-\u9fa5]{2,10})(?:旅游|旅行|玩|逛|攻略|行程|天气|景点|美食|路线|距离|吧|吗|呢|[，。！？\s]|$)"),
    re.compile(r"([\u4e00-\u9fa5]{2,10})(?:旅游|旅行)攻略"),
    re.compile(r"([\u4e00-\u9fa5]{2,10})\s*(?:\d+\s*(?:天|日)|几日游)"),
]
DAYS_RE = re.compile(r"(\d{1,2})\s*(?:天|日)")
PREF_MAP: dict[str, tuple[str, ...]] = {
    "美食": ("美食", "吃", "小吃", "餐厅", "火锅"),
    "自然": ("自然", "风景", "徒步", "山", "湖", "公园"),
    "人文": ("人文", "历史", "博物馆", "古城", "文化", "建筑"),
    "亲子": ("亲子", "小朋友", "孩子", "家庭"),
    "摄影": ("摄影", "拍照", "机位", "出片"),
    "休闲": ("轻松", "休闲", "慢游", "度假"),
}


def _human_texts(messages: list[BaseMessage]) -> list[str]:
    return [
        msg.content.strip()
        for msg in messages
        if isinstance(msg, HumanMessage) and (msg.content or "").strip()
    ]


def _missing_fields(city: str, days: int, preference: str) -> list[str]:
    missing: list[str] = []
    if not city:
        missing.append("city")
    if days <= 0:
        missing.append("days")
    if not preference:
        missing.append("preference")
    return missing


def _build_missing_prompt(missing: list[str]) -> str:
    readable = "、".join(
        MISSING_FIELD_LABELS[field] for field in missing if field in MISSING_FIELD_LABELS
    )
    return (
        f"为了给你生成准确的行程，还需要确认：**{readable}**。\n"
        "把这些信息告诉我后，我就可以继续规划。"
    )


def _normalize_intent(raw: str) -> str:
    intent = (raw or "").strip().lower()
    return intent if intent in VALID_INTENTS else "other"


def _sanitize_days(raw: object) -> int:
    try:
        days = int(raw)  # type: ignore[arg-type]
        return days if 1 <= days <= 30 else 0
    except (TypeError, ValueError):
        return 0


def _safe_parse_json(text: str) -> Optional[dict]:
    cleaned = re.sub(r"```(?:json)?\s*|```", "", text).strip()
    try:
        result = json.loads(cleaned)
        return result if isinstance(result, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _looks_like_profile_question(text: str) -> bool:
    text_lower = text.lower()
    return (
        any(keyword in text_lower for keyword in PROFILE_HINTS)
        and any(keyword in text for keyword in QUESTION_HINTS)
    )


def _looks_like_plan_request(text: str) -> bool:
    if not text:
        return False
    strong_patterns = (
        "帮我规划",
        "给我规划",
        "生成行程",
        "安排一下",
        "做个攻略",
        "旅游计划",
        "旅行计划",
    )
    return any(pattern in text for pattern in strong_patterns) or any(
        keyword in text for keyword in PLAN_HINTS
    )


def _regex_slots(text: str) -> tuple[str, int, str]:
    city = ""
    for pattern in CITY_PATTERNS:
        match = pattern.search(text)
        if match:
            city = match.group(1).strip()
            break

    days = 0
    match = DAYS_RE.search(text)
    if match:
        days = _sanitize_days(match.group(1))

    preferences = [
        label for label, keywords in PREF_MAP.items() if any(keyword in text for keyword in keywords)
    ]
    return city, days, "+".join(preferences)


def _regex_intent(latest: str, context: str) -> str:
    lower = latest.lower()

    if any(keyword in lower for keyword in CHAT_HINTS):
        return "general_chat"

    if _looks_like_profile_question(latest):
        return "need_answer"

    has_media = bool(MEDIA_RE.search(latest))
    has_answer = has_media or any(keyword in lower for keyword in ANSWER_HINTS)
    if any(keyword in latest for keyword in QUESTION_HINTS) and any(
        marker in latest
        for marker in ("天气", "景点", "美食", "餐厅", "路线", "距离", "交通", "偏好", "资料", "知识库")
    ):
        has_answer = True

    has_plan = _looks_like_plan_request(latest) or _looks_like_plan_request(context)

    if has_answer and not has_plan:
        return "need_answer"
    if has_plan:
        return "need_plan"
    if any(keyword in latest for keyword in QUESTION_HINTS):
        return "need_answer"
    return "other"


def _regex_classify(user_texts: list[str]) -> dict:
    recent = user_texts[-6:]
    latest = recent[-1]
    context = "\n".join(recent)

    city, days, preference = "", 0, ""
    for text in recent:
        city_new, days_new, pref_new = _regex_slots(text)
        city = city or city_new
        days = days or days_new
        preference = preference or pref_new

    city_new, days_new, pref_new = _regex_slots(latest)
    city = city_new or city
    days = days_new or days
    preference = pref_new or preference

    return {
        "intent": _regex_intent(latest, context),
        "city": city,
        "days": days,
        "preference": preference,
        "reason": "regex_fallback",
    }


def router_agent(state: TravelState) -> dict:
    messages: list[BaseMessage] = list(state.get("messages", []))
    user_texts = _human_texts(messages)
    user_input = user_texts[-1] if user_texts else ""

    if not user_input:
        return {
            "intent": "general_chat",
            "city": "",
            "days": 0,
            "preference": "",
            "missing_fields": [],
            "router_reason": "empty_input",
            "user_query": "",
            "messages": [
                AIMessage(
                    content="你好，告诉我你想去哪里、玩几天，以及更偏好的旅行风格，我就可以开始帮你规划。"
                )
            ],
        }

    recent_msgs = messages[-6:]
    context_lines = []
    for msg in recent_msgs[:-1]:
        if isinstance(msg, HumanMessage):
            context_lines.append(f"[用户] {msg.content}")
        elif isinstance(msg, AIMessage):
            context_lines.append(f"[AI] {msg.content}")
    context_for_llm = "\n".join(context_lines) + f"\n[当前] {user_input}"

    target_model = (state.get("router_model") or "glm-4-flash").strip()
    llm = None
    try:
        llm = get_llm(target_model)
    except Exception:
        pass

    parsed: Optional[dict] = None
    router_reason = "llm"

    if llm is not None:
        try:
            response = llm.invoke(
                [
                    SystemMessage(content=CLASSIFY_SYSTEM),
                    HumanMessage(content=context_for_llm),
                ]
            )
            raw_text = response.content if hasattr(response, "content") else str(response)
            parsed = _safe_parse_json(raw_text)
        except Exception:
            parsed = None

    if parsed is None:
        router_reason = "regex_fallback"
        parsed = _regex_classify(user_texts)

    state_city = (state.get("city") or "").strip()
    state_days = _sanitize_days(state.get("days", 0))
    state_preference = (state.get("preference") or "").strip()

    intent = _normalize_intent(str(parsed.get("intent", "")))
    city_new = str(parsed.get("city", "") or "").strip()
    days_new = _sanitize_days(parsed.get("days", 0))
    preference_new = str(parsed.get("preference", "") or "").strip()

    city = city_new or state_city
    days = days_new or state_days
    preference = preference_new or state_preference

    if _looks_like_profile_question(user_input):
        intent = "need_answer"

    if intent in {"general_chat", "other"} and not _missing_fields(city, days, preference):
        intent = "need_plan"

    missing: list[str] = []
    if intent in {"need_plan", "need_more_info"}:
        missing = _missing_fields(city, days, preference)
        intent = "need_more_info" if missing else "need_plan"

    reply: Optional[AIMessage] = None
    if intent == "need_more_info":
        reply = AIMessage(content=_build_missing_prompt(missing))
    elif intent in {"general_chat", "other"}:
        reply = AIMessage(
            content="我可以帮你做旅游规划，也可以回答景点、天气、路线、知识库内容等具体问题。"
        )

    output: dict = {
        "intent": intent,
        "city": city,
        "days": days,
        "preference": preference,
        "missing_fields": missing,
        "router_reason": router_reason,
        "user_query": user_input,
    }
    if reply is not None:
        output["messages"] = [reply]

    return output
