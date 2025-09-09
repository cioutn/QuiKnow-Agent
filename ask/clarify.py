from typing import List, Any, Dict
from logger import agent_logger
from .utils import extract_json

try:
    from langchain.schema import HumanMessage, SystemMessage  # type: ignore
except Exception:  # pragma: no cover
    HumanMessage = SystemMessage = None  # type: ignore


async def clarify_once(model, messages: List[Any]) -> Dict[str, Any]:
    last = messages[-1].content
    prompt = [
        SystemMessage(content=(
            "你是澄清助手。请将用户问题转为明确可检索表达，并判断是否需要追问。"
            "输出 JSON: {clarified:str, confirmed:bool, clarification_question:str|null, candidate_tags:[str]|null}")),
        HumanMessage(content=f"问题: {last}\n只输出 JSON")
    ]
    try:
        resp = await model.ainvoke(prompt)
        parsed = extract_json(resp.content)
        if not parsed:
            return {"clarified": resp.content, "confirmed": True, "clarification_question": None, "candidate_tags": []}
        clarified = parsed.get("clarified") or parsed.get("query") or last
        result = {
            "clarified": clarified,
            "confirmed": bool(parsed.get("confirmed", True)),
            "clarification_question": parsed.get("clarification_question"),
            "candidate_tags": parsed.get("candidate_tags") or []
        }
        agent_logger.debug(f"clarify -> {clarified} confirmed={result['confirmed']}")
        return result
    except Exception as e:  # pragma: no cover
        agent_logger.warning(f"clarify error: {e}")
        return {"clarified": last, "confirmed": True, "clarification_question": None, "candidate_tags": []}
