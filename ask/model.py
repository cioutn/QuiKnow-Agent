import settings
from logger import agent_logger

try:
    from langchain_openai import ChatOpenAI  # type: ignore
except Exception:
    ChatOpenAI = None  # type: ignore
try:
    from langchain_anthropic import ChatAnthropic  # type: ignore
except Exception:
    ChatAnthropic = None  # type: ignore
try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
except Exception:
    ChatGoogleGenerativeAI = None  # type: ignore


def init_chat_model(model_name: str | None) -> object:
    name = model_name or settings.MODEL_NAME
    protocol = settings.MODEL_PROTOCOL.upper()
    try:
        if protocol == "OPENAI" and ChatOpenAI:
            base = settings.MODEL_URL or ""
            # 补协议，去重末尾斜杠
            if not base.startswith(("http://", "https://")):
                base = "http://" + base
            base = base.rstrip("/")
            # 本地 ollama 特殊处理（默认 127.0.0.1:11434）
            if "127.0.0.1:11434" in base or "localhost:11434" in base:
                return ChatOpenAI(model=name, openai_api_base="http://127.0.0.1:11434/v1", openai_api_key="ollama")
            return ChatOpenAI(model=name, openai_api_base=base, openai_api_key=settings.MODEL_KEY or "dummy-key")
        if protocol == "ANTHROPIC" and ChatAnthropic:
            return ChatAnthropic(model=name, anthropic_api_key=settings.MODEL_KEY)
        if protocol == "GOOGLE" and ChatGoogleGenerativeAI:
            return ChatGoogleGenerativeAI(model=name, google_api_key=settings.MODEL_KEY)
    except Exception as e:  # pragma: no cover
        agent_logger.warning(f"model init failed: {e}")
    # fallback
    if ChatOpenAI:
        agent_logger.info("fallback: local ollama openai adapter")
        return ChatOpenAI(model=name, openai_api_base="http://127.0.0.1:11434/v1", openai_api_key="ollama")
    raise RuntimeError("No available chat model backend installed")
