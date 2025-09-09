import asyncio, json, re
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List
import settings
from logger import agent_logger
from .model import init_chat_model
from .sql import maybe_sql_analyze
from .metrics import estimate_tokens, summarize_blocks

try:
    from langchain.schema import HumanMessage, SystemMessage  # type: ignore
except Exception:  # pragma: no cover
    HumanMessage = SystemMessage = None  # type: ignore

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient  # type: ignore
except Exception:  # pragma: no cover
    MultiServerMCPClient = None  # type: ignore


class AskAgent:
    """三阶段检索回答：overview -> structure -> gather leaves."""

    def __init__(self, model, server_spec: dict, tool_timeout: float):
        self.model = model
        self.server_spec = server_spec
        self.tool_timeout = tool_timeout
        self.mcp_client = None
        self.tools: List[Any] = []

    async def _init_client(self):
        self.mcp_client = MultiServerMCPClient(self.server_spec)
        self.tools = await self.mcp_client.get_tools()

    @classmethod
    async def create(cls, model_name: str | None = None, mcp_name: str = "QuiKnow"):
        model = init_chat_model(model_name)
        if not MultiServerMCPClient:
            raise RuntimeError("langchain_mcp_adapters 未安装")
        spec = {mcp_name: {"transport": "streamable_http", "url": f"http://{settings.MCP_HOST}:{settings.MCP_PORT}{settings.MCP_PATH}"}}
        self = cls(model, spec, settings.TOOL_TIMEOUT)
        await self._init_client()
        async def _sql_exec(payload: dict):
            return await self.call_tool("sql_tool", payload)
        self.sql_executor = _sql_exec  # type: ignore[attr-defined]
        return self

    def find_tool(self, name: str):
        for t in self.tools:
            if getattr(t, "name", None) == name or getattr(t, "__name__", None) == name:
                return t
        return None

    async def call_tool(self, tool_name: str, params: dict) -> dict:
        tool = self.find_tool(tool_name)
        if not tool:
            raise RuntimeError(f"tool {tool_name} not found")
        async def _invoke():
            for attr in ("acall", "ainvoke", "arun"):
                if hasattr(tool, attr):
                    return await getattr(tool, attr)(params)
            loop = asyncio.get_running_loop()
            if hasattr(tool, "call"):
                return await loop.run_in_executor(None, lambda: tool.call(params))
            if hasattr(tool, "invoke"):
                return await loop.run_in_executor(None, lambda: tool.invoke(params))
            if hasattr(tool, "run"):
                return await loop.run_in_executor(None, lambda: tool.run(params))
            if callable(tool):
                if asyncio.iscoroutinefunction(tool):  # type: ignore
                    return await tool(params)  # type: ignore
                return await loop.run_in_executor(None, lambda: tool(params))  # type: ignore
            raise RuntimeError("unknown tool interface")
        try:
            raw = await asyncio.wait_for(_invoke(), timeout=self.tool_timeout)
        except Exception as e:
            agent_logger.error(f"调用工具失败 {tool_name}: {e}")
            return {"status": "error", "message": str(e)}
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
                return {"status": "success", "result": parsed}
            except Exception:
                return {"status": "raw", "result": raw}
        return {"status": "success", "result": raw}

    async def get_tags(self, question: str) -> List[str]:
        base = question.strip()
        if not base:
            return []
        heuristic = [w for w in re.split(r"[\s,，。；;:/]+", base) if w and 1 < len(w) <= 40][:12]
        if HumanMessage and SystemMessage:
            prompt = [
                SystemMessage(content="从问题中提取 3-8 个检索关键词或短语，逗号分隔，只输出结果。保持专有名词原样。"),
                HumanMessage(content=base)
            ]
            try:
                resp = await self.model.ainvoke(prompt)
                model_tags = [t.strip() for t in resp.content.replace("\n", " ").split(",") if t.strip()]
                if model_tags:
                    heuristic.extend(model_tags)
            except Exception:
                pass
        seen, final = set(), []
        for t in heuristic:
            if t.lower() not in seen and len(final) < 12:
                seen.add(t.lower())
                final.append(t)
        return final

    async def ask(self, question: str, write_board: bool = True) -> Dict[str, Any]:
        agent_logger.info(f"处理问题: {question}")
        tags = await self.get_tags(question)
        agent_logger.debug(f"检索 tags: {tags}")

        overview = await self.call_tool("search_documents", {"mode": "overview", "keywords": tags})
        if overview.get("status") != "success":
            return {"answer": "目录检索失败", "final_context": "", "tags": tags}
        tree_text = overview.get("result", "")
        agent_logger.debug(f"目录树长度: {len(tree_text)}")

        if HumanMessage and SystemMessage:
            sel_prompt = [
                SystemMessage(content="这是知识库目录树，含 #id 与 [HIT] 标记。只输出最相关 1-5 个文件ID，逗号分隔。"),
                HumanMessage(content=f"问题: {question}\n目录:\n{tree_text}")
            ]
            try:
                resp = await self.model.ainvoke(sel_prompt)
                raw_ids = resp.content.strip().split(',')
                file_ids = [re.sub(r'#id:?\s*', '', r.strip()) for r in raw_ids if r.strip()][:5]
            except Exception:
                file_ids = []
        else:
            file_ids = []
        if not file_ids:
            return {"answer": "未选出相关文件", "final_context": tree_text, "tags": tags}
        agent_logger.info(f"选中文件: {file_ids}")

        expanded = await self.call_tool("search_documents", {"mode": "expand", "ids": file_ids, "keywords": tags})
        if expanded.get("status") != "success":
            return {"answer": "文件展开失败", "final_context": tree_text, "tags": tags, "chosen_files": file_ids}
        expanded_text = expanded.get("result", "")
        agent_logger.debug(f"结构展开长度: {len(expanded_text)}")

        if HumanMessage and SystemMessage:
            struct_prompt = [
                SystemMessage(content=(
                    "下面是目标文件内部结构(内容叶子已隐藏；若为 CSV/Excel 会附加 preview 节点)。\n"
                    "任务: 选择最相关 1-6 个最小结构节点ID。\n"
                    "规则: 若问题与某数据文件(.csv/.xls/.xlsx)内容或统计相关, 必须直接包含该文件的 type=0 节点ID (不要只选预览子节点)。\n"
                    "输出格式: NODES: id1,id2,... 只输出这一行。")),
                HumanMessage(content=f"问题: {question}\n结构树:\n{expanded_text}")
            ]
            try:
                s_resp = await self.model.ainvoke(struct_prompt)
                line = s_resp.content.strip().splitlines()[0]
                m = re.search(r"NODES?:\s*(.+)", line, re.I)
                if m:
                    raw_node_ids = [x.strip() for x in m.group(1).split(',') if x.strip()]
                else:
                    raw_node_ids = []
            except Exception:
                raw_node_ids = []
        else:
            raw_node_ids = []
        struct_ids = raw_node_ids[:6] if raw_node_ids else file_ids[:2]
        agent_logger.info(f"选中结构节点: {struct_ids}")
        gather = await self.call_tool("gather_context", {"node_ids": struct_ids, "keywords": tags})
        leaves_block = ""
        leaf_contexts: List[Dict[str, Any]] = []
        if gather.get("status") == "success":
            nodes_data = gather.get("nodes", [])
            out_chunks = []
            for nd in nodes_data:
                for lf in nd.get('leaves', []):
                    leaf_contexts.append({
                        'id': lf.get('id'), 'context': lf.get('context'),
                        'node_type': lf.get('node_type', 'leaf'), 'hit': lf.get('hit')
                    })
                    out_chunks.append(f"# Leaf {lf.get('id')} {'[HIT]' if lf.get('hit') else ''}\n{lf.get('context')}")
            leaves_block = "\n\n".join(out_chunks)
        else:
            agent_logger.warning(f"gather_context 失败: {gather}")

        sql_extra_obj = await maybe_sql_analyze(leaf_contexts, question, getattr(self, 'sql_executor', None), self.model)
        sql_extra_text = ""
        if isinstance(sql_extra_obj, dict):
            if sql_extra_obj.get("mode") == "sql":
                sql_extra_text = f"执行SQL: {sql_extra_obj.get('sql')}\n结果: {json.dumps(sql_extra_obj.get('sql_result'), ensure_ascii=False)[:2000]}"
            elif sql_extra_obj.get("mode") == "nl":
                sql_extra_text = f"结构化分析: {sql_extra_obj.get('answer','')}"
        context_block = expanded_text
        if leaves_block:
            context_block += "\n\n--- LEAVES ---\n" + leaves_block
        if sql_extra_text:
            context_block += "\n\n--- SQL ---\n" + sql_extra_text

        metrics = {
            'expanded_chars': len(expanded_text),
            'expanded_tokens': estimate_tokens(expanded_text, settings.MODEL_NAME),
            'leaf_stats': summarize_blocks([c['context'] for c in leaf_contexts], settings.MODEL_NAME),
            'sql_chars': len(sql_extra_text),
            'sql_tokens': estimate_tokens(sql_extra_text, settings.MODEL_NAME)
        }

        if HumanMessage and SystemMessage:
            ans_prompt = [
                SystemMessage(content="根据上下文回答问题。若不确定明确说明。不虚构。"),
                HumanMessage(content=f"问题: {question}\n上下文:\n{context_block}")
            ]
            try:
                ans_resp = await self.model.ainvoke(ans_prompt)
                answer = ans_resp.content
            except Exception as e:
                answer = f"回答失败: {e}"
        else:
            answer = "模型不可用"

        if write_board:
            try:
                board_path = Path(__file__).resolve().parent.parent / "ask.md"
                board_content = (
                    f"# 最新问答\n\n" \
                    f"**问题**\n\n{question}\n\n" \
                    f"**回答**\n\n{answer}\n" \
                )
                board_path.write_text(board_content, encoding='utf-8')
            except Exception as e:  # 不影响主流程
                agent_logger.warning(f"写入 ask.md 失败: {e}")

        return {
            "answer": answer,
            "final_context": context_block,
            "tags": tags,
            "chosen_files": file_ids,
            "structure_nodes": struct_ids,
            "gathered_leaf_count": len(leaf_contexts),
            "metrics": metrics
        }

    async def ask_interactive(self, question: str, user_response_callback: Callable[[str], Awaitable[str]]):  # 兼容 CLI 调用
        return await self.ask(question)

    async def ask_report(self, question: str, max_sub_questions: int = 5) -> Dict[str, Any]:
        """多轮分解汇总报告模式。"""
        if not question.strip():
            return {"mode": "report", "error": "空问题"}
        sub_questions: List[str] = []
        if HumanMessage and SystemMessage:
            prompt = [
                SystemMessage(content=(
                    "将用户问题分解为 3-6 个互补子问题，覆盖不同方面。\n"
                    "严格输出 JSON 数组(仅字符串)，不要代码块、不要对象、不要多余文字。每项<=40字。示例: [\"子问题1\", \"子问题2\"]")),
                HumanMessage(content=question)
            ]
            try:
                resp = await self.model.ainvoke(prompt)
                txt = resp.content.strip()
                # 去除代码块围栏
                if txt.startswith("```"):
                    # 删除首行 ```lang 和末尾 ```
                    lines = [l for l in txt.splitlines() if not l.strip().startswith("```")]
                    txt = "\n".join(lines).strip()
                import json as _j, re as _re
                parsed = None
                # 直接尝试解析
                try:
                    parsed = _j.loads(txt)
                except Exception:
                    # 如果包含类似 "question": 结构，尝试构造一个数组
                    if '"question"' in txt:
                        # 提取 "question": "xxx" 片段
                        cand = _re.findall(r'"question"\s*:\s*"(.+?)"', txt)
                        if cand:
                            parsed = cand
                if isinstance(parsed, dict) and 'questions' in parsed:
                    parsed = parsed['questions']
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, str):
                            s = item.strip().strip('"')
                            if s:
                                sub_questions.append(s)
                        elif isinstance(item, dict):
                            qv = item.get('question') or item.get('q')
                            if isinstance(qv, str) and qv.strip():
                                sub_questions.append(qv.strip())
                if not sub_questions:
                    # 逐行 fallback
                    raw_lines = [l.strip() for l in txt.splitlines() if l.strip()]
                    cleaned = []
                    skip_prefix = {"[", "]", "{", "}", "-", "*"}
                    for ln in raw_lines:
                        if any(ln.startswith(p) and len(ln) <= 3 for p in skip_prefix):
                            continue
                        if ln.lower() in {"json", "array"}:
                            continue
                        if ln.startswith(('"', "'")) and ln.endswith(('"', "'")):
                            ln = ln[1:-1]
                        ln = _re.sub(r'^[-*]\s*', '', ln)
                        ln = _re.sub(r'^"?question"?\s*:\s*', '', ln, flags=_re.I)
                        ln = ln.rstrip(',')
                        if 2 <= len(ln) <= 80 and not all(ch in '[]{}:,"' for ch in ln):
                            cleaned.append(ln)
                    sub_questions.extend(cleaned)
            except Exception:
                pass
        if not sub_questions:
            sub_questions = [question]
        sub_questions = sub_questions[:max_sub_questions]

        sub_results = []
        for sq in sub_questions:
            r = await self.ask(sq, write_board=False)
            sub_results.append({
                'question': sq,
                'answer': r.get('answer'),
                'metrics': r.get('metrics'),
                'chosen_files': r.get('chosen_files'),
                'structure_nodes': r.get('structure_nodes')
            })

        # 汇总报告
        combined_answer = ''
        if HumanMessage and SystemMessage:
            syntho = [
                SystemMessage(content="汇总子问题答案，生成结构化报告：\n1. 总结\n2. 关键发现\n3. 数据/证据引用(若有)\n4. 风险或不确定性\n5. 后续建议。"),
                HumanMessage(content=json.dumps({
                    'main_question': question,
                    'sub_results': sub_results
                }, ensure_ascii=False)[:12000])
            ]
            try:
                resp = await self.model.ainvoke(syntho)
                combined_answer = resp.content
            except Exception as e:
                combined_answer = f"汇总失败: {e}"
        else:
            combined_answer = "模型不可用"

        # 写入 report.md
        try:
            report_path = Path(__file__).resolve().parent.parent / 'report.md'
            report_path.write_text(
                f"# 报告\n\n**主问题**\n\n{question}\n\n**子问题**\n\n" + '\n'.join(f"- {sq}" for sq in sub_questions) +
                "\n\n**最终报告**\n\n" + combined_answer + "\n",
                encoding='utf-8'
            )
        except Exception as e:
            agent_logger.warning(f"写入 report.md 失败: {e}")

        return {
            'mode': 'report',
            'main_question': question,
            'sub_questions': sub_questions,
            'sub_results': sub_results,
            'report': combined_answer
        }

