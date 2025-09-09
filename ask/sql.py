import json
from typing import List, Dict, Any, Optional
from logger import sql_logger

try:
    from langchain.schema import SystemMessage, HumanMessage  # type: ignore
except Exception:  # pragma: no cover
    SystemMessage = HumanMessage = None  # type: ignore

FORBIDDEN_SQL_TOKENS = [";", "--", "/*", " attach ", " pragma ", " drop ", " delete ", " update ", " insert ", " alter ", " create ", " replace ", " vacuum "]


def sanitize_sql(sql: str) -> str:
    s = sql.strip().strip(";")
    low = s.lower()
    if not low.startswith("select"):
        raise ValueError("only SELECT allowed")
    for kw in FORBIDDEN_SQL_TOKENS:
        if kw.strip() in low and (kw.startswith(" ") or kw.endswith(" ") or kw in [";","--","/*"]):
            raise ValueError(f"forbidden token: {kw}")
    if " limit " not in low:
        s += " LIMIT 200"
    if len(s) > 2000:
        raise ValueError("sql too long")
    return s


async def maybe_sql_analyze(contexts: List[dict], question: str, sql_executor=None, model=None) -> Optional[Dict[str, Any]]:
    """统一 SQL 决策入口：模型决定是否执行 SQL。"""
    if not model:
        return None

    csv_keywords = ["csv", "表", "字段", "列", "数据", "统计"]
    lower_q = question.lower()
    def has_csv_kw():
        return any(kw in lower_q for kw in csv_keywords)

    csv_ctx = [c for c in contexts if c.get("node_type") == "csv_excel"]
    sql_logger.debug(f"maybe_sql_analyze: csv_excel contexts={len(csv_ctx)} question={question[:80]}")
    if not csv_ctx and has_csv_kw():
        csv_ctx = [{"id": "csv_fallback", "context": "(无预览数据，可能需要列名与样例行)", "node_type": "csv_excel"}]
    if not csv_ctx:
        return None

    parsed_items = []
    for item in csv_ctx[:2]:
        ctx_text = item.get("context", "")
        schema_part = ""
        sample_part = ""
        if "SCHEMA:" in ctx_text:
            parts = ctx_text.split("SCHEMA:", 1)[1]
            if "SAMPLE:" in parts:
                schema_part, sample_part = parts.split("SAMPLE:", 1)
            else:
                schema_part = parts
        parsed_items.append({
            "id": item.get("id"),
            "schema": schema_part.strip()[:800],
            "sample": sample_part.strip()[:800]
        })

    instruction = (
        "你是数据分析助手。判断是否需要通过 SQL(只读) 查询来回答问题。\n"
        "如果 SQL 能显著提升准确性或需要聚合/过滤/统计，则输出 JSON: {\"mode\":\"sql\",\"sql\":\"SELECT ...\"}.\n"
        "否则输出 JSON: {\"mode\":\"nl\",\"answer\":\"直接答案\"}.\n"
        "要求: 1) 只能 SELECT；2) 必须含 LIMIT (<=200)；3) 不猜测不存在列；4) 如列不完整先 NL 说明需要更多上下文。\n"
        "示例1: {\"mode\":\"sql\",\"sql\":\"SELECT col1, COUNT(*) c FROM data LIMIT 50\"}\n"
        "示例2: {\"mode\":\"nl\",\"answer\":\"该文件只有列A,B...\"}"
    )
    user_block = {"question": question, "files": parsed_items}

    from .utils import extract_json
    try:
        resp = await model.ainvoke([
            SystemMessage(content=instruction),
            HumanMessage(content=json.dumps(user_block, ensure_ascii=False))
        ])
        parsed = extract_json(resp.content)
        if not parsed:
            return None
        mode = (parsed.get("mode") or "").lower()
        sql_logger.debug(f"maybe_sql_analyze: model decision raw={parsed}")
        if mode == "sql":
            raw_sql = parsed.get("sql", "")
            try:
                safe_sql = sanitize_sql(raw_sql)
            except Exception as e:
                sql_logger.warning(f"sql rejected: {e} sql={raw_sql}")
                return {"mode": "nl", "answer": "（SQL 不安全或无效，改为自然语言）"}
            exec_fn = sql_executor or getattr(model, "agent_call_sql", None)
            if not exec_fn:
                return {"mode": "nl", "answer": "（系统缺少 SQL 执行能力）"}
            sql_logger.info(f"exec sql: {safe_sql}")
            res = await exec_fn({"sql": safe_sql})  # type: ignore
            return {"mode": "sql", "sql": safe_sql, "sql_result": res}
        return {"mode": "nl", "answer": parsed.get("answer", "")}
    except Exception as e:  # pragma: no cover
        sql_logger.warning(f"sql analyze error: {e}")
        return None
