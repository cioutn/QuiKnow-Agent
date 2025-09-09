from decouple import config, Csv
from pathlib import Path

# MCP 服务器配置
MCP_HOST = config('MCP_HOST', default="127.0.0.1")
MCP_PORT = config('MCP_PORT', default=9000, cast=int)
MCP_PATH = config('MCP_PATH', default="/mcp")

# 模型配置
MODEL_PROTOCOL = config('MODEL_PROTOCOL', default="OPENAI")
MODEL_URL = config('MODEL_URL', default="127.0.0.1:11434/v1")
MODEL_NAME = config('MODEL_NAME', default="qwen3:0.6b")
MODEL_KEY = config('MODEL_KEY', default="")

# 调试配置
DEBUG = config('DEBUG', default=False, cast=bool)

# 工具超时配置
TOOL_TIMEOUT = config('TOOL_TIMEOUT', default=8.0, cast=float)

# 搜索配置
MAX_REFINE_ITERATIONS = config('MAX_REFINE_ITERATIONS', default=6, cast=int)
CANDIDATE_TARGET = config('CANDIDATE_TARGET', default=6, cast=int)

# 日志配置
LOG_LEVEL = config('LOG_LEVEL', default="INFO")
LOG_TO_FILE = config('LOG_TO_FILE', default=True, cast=bool)
LOG_DIR = config('LOG_DIR', default="logs/")
