import json, re
from typing import List, Dict, Any, Optional, Tuple
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
    detected_tables = set()
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
        # 捕获表名：优先 TABLE: ，其次 对应数据表:
        table_name = None
        m1 = re.search(r'(?:^|\n)TABLE:\s*([A-Za-z0-9_]+)', ctx_text)
        if m1:
            table_name = m1.group(1)
        else:
            m2 = re.search(r'对应数据表:\s*([A-Za-z0-9_]+)', ctx_text)
            if m2:
                table_name = m2.group(1)
        if table_name:
            detected_tables.add(table_name)
        parsed_items.append({
            "id": item.get("id"),
            "schema": schema_part.strip()[:800],
            "sample": sample_part.strip()[:800],
            **({"table": table_name} if table_name else {})
        })

    tables_list = sorted(detected_tables)

    # --- 方案2: 获取列信息并区分需要引号的列 ---
    columns_meta: Dict[str, Dict[str, List[str]]] = {}
    exec_fn = sql_executor or getattr(model, "agent_call_sql", None)
    if exec_fn:
        for tb in tables_list:
            try:
                pragma_sql = f'PRAGMA table_info("{tb}")'
                r = await exec_fn({"sql": pragma_sql})  # type: ignore
                cols = []
                if isinstance(r, dict) and r.get("status") == "success":
                    rows = r.get("rows") or []
                    for row in rows:
                        # sqlite pragma columns: cid, name, type, notnull, dflt_value, pk
                        name = row.get("name") if isinstance(row, dict) else None
                        if name:
                            cols.append(name)
                safe_cols = []
                quote_cols = []
                for c in cols:
                    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", c or ""):
                        safe_cols.append(c)
                    else:
                        quote_cols.append(c)
                columns_meta[tb] = {"safe": safe_cols, "quote": quote_cols}
            except Exception:
                continue

    # 构建提示文本
    tables_clause = ("可用数据表: " + ", ".join(tables_list)) if tables_list else "未检测到可用表名"
    quote_advice_lines = []
    for tb in tables_list:
        meta = columns_meta.get(tb, {})
        qcols = meta.get("quote", [])
        if qcols:
            quote_advice_lines.append(f"表 {tb} 中需加引号列: " + ", ".join(f'"{c}"' for c in qcols))
    quote_advice = ("\n" + "\n".join(quote_advice_lines)) if quote_advice_lines else ""

    instruction = (
        "你是数据分析助手。判断是否需要通过只读 SQL 查询来回答问题。\n"
        f"{tables_clause}{quote_advice}\n"
        "若生成 SQL:\n"
        "  - 只能使用上述真实表名；禁止使用 data（除非列在列表中）。\n"
        "  - 必须 SELECT 且包含 LIMIT。\n"
        "  - 对包含特殊字符(非字母数字下划线或以数字开头)的列，必须使用双引号。\n"
        "输出 JSON 之一： {\"mode\":\"sql\",\"sql\":\"...\"} 或 {\"mode\":\"nl\",\"answer\":\"...\"}.\n"
        "示例_SQL: {\"mode\":\"sql\",\"sql\":\"SELECT Product_Category, SUM(Revenue_USD) AS Total_Revenue FROM Quarterly_Earnings_Q4_2022 GROUP BY Product_Category LIMIT 50\"}\n"
        "示例_NL: {\"mode\":\"nl\",\"answer\":\"列信息不足，需更多上下文\"}"
    )
    user_block = {"question": question, "files": parsed_items}
    if tables_list:
        user_block["tables"] = tables_list
        user_block["columns_meta"] = columns_meta

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
            # 若使用了 data 且 data 不在允许表中，尝试自动替换或拒绝
            low = raw_sql.lower()
            if " from data" in low and ("data" not in tables_list):
                if len(tables_list) == 1:
                    pattern = re.compile(r'\bdata\b', re.IGNORECASE)
                    raw_sql = pattern.sub(tables_list[0], raw_sql)
                    sql_logger.debug(f"auto replace table name -> {raw_sql}")
                else:
                    return {"mode": "nl", "answer": "需要使用真实表名(非 data)。请重试。"}

            def auto_quote(sql_text: str, meta: Dict[str, Dict[str, List[str]]]) -> str:
                out = sql_text
                for tb, info in meta.items():
                    for col in info.get("quote", []):
                        # 仅替换未被双引号包裹的裸出现
                        # 构造前后边界：非字母数字下划线或行首/行尾
                        pattern = re.compile(rf'(?<!["])(?<![A-Za-z0-9_]){re.escape(col)}(?![A-Za-z0-9_])(?!["])')
                        out_new = pattern.sub(f'"{col}"', out)
                        out = out_new
                return out

            try:
                safe_sql = sanitize_sql(raw_sql)
            except Exception as e:
                sql_logger.warning(f"sql rejected: {e} sql={raw_sql}")
                return {"mode": "nl", "answer": "（SQL 不安全或无效，改为自然语言）"}

            # --- 方案1: 执行前自动加引号 ---
            if columns_meta:
                quoted_sql = auto_quote(safe_sql, columns_meta)
            else:
                quoted_sql = safe_sql

            exec_fn2 = sql_executor or getattr(model, "agent_call_sql", None)
            if not exec_fn2:
                return {"mode": "nl", "answer": "（系统缺少 SQL 执行能力）"}
            sql_logger.info(f"exec sql: {quoted_sql}")
            res = await exec_fn2({"sql": quoted_sql})  # type: ignore

            # --- 方案3: 错误兜底修复重试 ---
            if isinstance(res, dict) and res.get("status") == "error" and isinstance(res.get("message"), str):
                msg = res.get("message", "")
                m_err = re.search(r"no such column: ([A-Za-z0-9_]+)", msg)
                if m_err and columns_meta:
                    missing = m_err.group(1)
                    # 尝试找到以 missing + 特殊字符开头的实际列（如 R 对应 R&D_Spend_USD）
                    repair_candidates: List[Tuple[str,str]] = []  # (table, col)
                    for tb, info in columns_meta.items():
                        for col in info.get("quote", []) + info.get("safe", []):
                            if col.startswith(missing) and len(col) > len(missing):
                                repair_candidates.append((tb, col))
                    if len(repair_candidates) == 1:
                        _, target_col = repair_candidates[0]
                        # 再次强制替换该列所有裸片段（可能被拆分）
                        forced_pattern = re.compile(rf'(?<!["]){re.escape(target_col.split("&")[0])}(?=[&])')
                        repaired_sql = forced_pattern.sub(f'"{target_col}"', quoted_sql)
                        sql_logger.debug(f"repair retry sql: {repaired_sql}")
                        res2 = await exec_fn2({"sql": repaired_sql})  # type: ignore
                        return {"mode": "sql", "sql": repaired_sql, "sql_result": res2}
            return {"mode": "sql", "sql": quoted_sql, "sql_result": res}
        return {"mode": "nl", "answer": parsed.get("answer", "")}
    except Exception as e:  # pragma: no cover
        sql_logger.warning(f"sql analyze error: {e}")
        return None
