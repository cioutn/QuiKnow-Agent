"""项目主入口 CLI。

子命令：
  ask                进入交互问答（默认）
  build [--path P]   启动文档/数据构建任务（调用 start_document_build）
  status --job ID    查询构建任务状态（调用 get_job_status）
  tree               触发目录聚类建树（调用 directory_tree_builder）

示例：
  python main.py ask
  python main.py build --path ./docs
  python main.py status --job 123e4567-...
  python main.py tree
"""
import asyncio, json, argparse, sys, inspect, time
from ask import AskAgent
from ask.model import init_chat_model
import settings

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient  # type: ignore
except Exception:  # pragma: no cover
    MultiServerMCPClient = None  # type: ignore


async def _cli_callback(prompt: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: input(prompt + "\n> "))


def _build_spec():
    return {
        "QuiKnow": {
            "transport": "streamable_http",
            "url": f"http://{settings.MCP_HOST}:{settings.MCP_PORT}{settings.MCP_PATH}",
        }
    }


async def _get_tool(name: str):
    if not MultiServerMCPClient:
        raise RuntimeError("langchain_mcp_adapters 未安装")
    client = MultiServerMCPClient(_build_spec())
    tools = await client.get_tools()
    for t in tools:
        if getattr(t, "name", None) == name:
            return t
    raise RuntimeError(f"未找到工具: {name}")


async def _invoke_tool(tool, payload: dict):
    """通用工具调用：兼容 acall / ainvoke / arun / call / invoke / run。"""
    # 优先异步专用接口
    for attr in ("acall", "ainvoke", "arun"):
        if hasattr(tool, attr):
            return await getattr(tool, attr)(payload)
    loop = asyncio.get_running_loop()
    if hasattr(tool, "call"):
        return await loop.run_in_executor(None, lambda: tool.call(payload))
    if hasattr(tool, "invoke"):
        return await loop.run_in_executor(None, lambda: tool.invoke(payload))
    if hasattr(tool, "run"):
        return await loop.run_in_executor(None, lambda: tool.run(payload))
    if callable(tool):
        if inspect.iscoroutinefunction(tool):  # type: ignore
            return await tool(payload)  # type: ignore
        return await loop.run_in_executor(None, lambda: tool(payload))  # type: ignore
    raise RuntimeError("无法调用该工具：未知接口")


async def cmd_check(args):
    """精简版健康检查：只做一次模型初始化+简单调用。"""
    try:
        from langchain.schema import HumanMessage, SystemMessage  # type: ignore
    except Exception:  # pragma: no cover
        HumanMessage = SystemMessage = None  # type: ignore
    model_name = args.model or None
    start = time.perf_counter()
    out: dict[str, any] = {
        "model": model_name or settings.MODEL_NAME,
        "protocol": settings.MODEL_PROTOCOL,
        "base_url": settings.MODEL_URL,
    }
    try:
        model = init_chat_model(model_name)
        msg = [SystemMessage(content="健康检查"), HumanMessage(content="只回复 OK")] if HumanMessage and SystemMessage else []
        resp = None
        if hasattr(model, "ainvoke"):
            resp = await model.ainvoke(msg)
        elif hasattr(model, "invoke"):
            resp = await asyncio.get_running_loop().run_in_executor(None, lambda: model.invoke(msg))  # type: ignore
        elif hasattr(model, "apredict"):
            resp = await model.apredict("OK")  # type: ignore
        elif hasattr(model, "predict"):
            resp = await asyncio.get_running_loop().run_in_executor(None, lambda: model.predict("OK"))  # type: ignore
        content = getattr(resp, "content", None) or getattr(resp, "text", None) or str(resp)
        out.update({
            "status": "success",
            "latency_sec": round(time.perf_counter() - start, 3),
            "reply": (content or "")[:200],
            "ok": "OK" in (content or "").upper(),
        })
    except Exception as e:
        out.update({
            "status": "error",
            "error": str(e),
            "latency_sec": round(time.perf_counter() - start, 3),
        })
    print(json.dumps(out, ensure_ascii=False, indent=2))


async def cmd_ask(args):
    print("交互问答模式（空行退出）")
    agent = await AskAgent.create()
    while True:
        q = input("问题> ").strip()
        if not q:
            break
        res = await agent.ask_interactive(q, user_response_callback=_cli_callback)
        print(json.dumps(res, ensure_ascii=False, indent=2))


async def cmd_build(args):
    tool = await _get_tool("start_document_build")
    payload = {"file_path": args.path} if args.path else {}
    res = await _invoke_tool(tool, payload)
    print(json.dumps(res, ensure_ascii=False, indent=2))

async def cmd_report(args):
    agent = await AskAgent.create()
    question = " ".join(args.question).strip()
    res = await agent.ask_report(question, max_sub_questions=args.max_sub)
    print(json.dumps(res, ensure_ascii=False, indent=2))


async def cmd_status(args):
    tool = await _get_tool("get_job_status")
    res = await _invoke_tool(tool, {"job_id": args.job})
    print(json.dumps(res, ensure_ascii=False, indent=2))


async def cmd_tree(args):
    tool = await _get_tool("directory_tree_builder")
    res = await _invoke_tool(tool, {})
    print(json.dumps(res, ensure_ascii=False, indent=2))


def parse_args(argv: list[str]):
    parser = argparse.ArgumentParser(description="QuiKnow CLI")
    sub = parser.add_subparsers(dest="command")

    p_ask = sub.add_parser("ask", help="进入问答模式")
    p_ask.set_defaults(func=cmd_ask)

    p_build = sub.add_parser("build", help="启动文档/数据构建")
    p_build.add_argument("--path", type=str, help="文件或目录路径", required=False)
    p_build.set_defaults(func=cmd_build)

    p_status = sub.add_parser("status", help="查询构建任务状态")
    p_status.add_argument("--job", type=str, required=True, help="任务ID")
    p_status.set_defaults(func=cmd_status)

    p_tree = sub.add_parser("tree", help="触发目录聚类建树")
    p_tree.set_defaults(func=cmd_tree)

    p_check = sub.add_parser("check", help="验证模型/API Key 可用性")
    p_check.add_argument("--model", type=str, required=False, help="显式指定模型名称（否则使用默认配置）")
    p_check.set_defaults(func=cmd_check)

    p_report = sub.add_parser("report", help="多轮分解并生成报告")
    p_report.add_argument("question", nargs="+", help="主问题文本")
    p_report.add_argument("--max-sub", type=int, default=5, help="最多子问题数 (默认5)")
    p_report.set_defaults(func=cmd_report)

    return parser.parse_args(argv)


def main(argv: list[str]):
    args = parse_args(argv)
    if not args.command:
        # 默认进入 ask
        args.command = "ask"
        args.func = cmd_ask  # type: ignore
    asyncio.run(args.func(args))  # type: ignore


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
