import logging
import os
from pathlib import Path
from typing import Optional
import settings

class Logger:
    """统一的日志工具类"""
    
    def __init__(self, name: str, log_dir: Optional[str] = None):
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else Path(settings.LOG_DIR)
        self.logger = logging.getLogger(name)
        self.setup_logger()
    
    def setup_logger(self):
        """设置日志器配置"""
        # 清除已有的处理器
        self.logger.handlers.clear()
        
        # 设置日志级别
        log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
        self.logger.setLevel(log_level)
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 添加控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 如果需要记录到文件
        if settings.LOG_TO_FILE:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_file = self.log_dir / f"{self.name}.log"
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str):
        """记录DEBUG级别日志"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """记录INFO级别日志"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """记录WARNING级别日志"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """记录ERROR级别日志"""
        self.logger.error(message)
    
    def exception(self, message: str):
        """记录异常信息"""
        self.logger.exception(message)

agent_logger = Logger("agent")
sql_logger = Logger("sql")  # 目前仅这两个在代码中实际使用

_DYNAMIC_CACHE: dict[str, Logger] = {}

def get_logger(name: str) -> Logger:
    """按需创建其他分类日志，避免生成空日志文件。"""
    if name in ("agent", "sql"):
        return agent_logger if name == "agent" else sql_logger
    if name not in _DYNAMIC_CACHE:
        _DYNAMIC_CACHE[name] = Logger(name)
    return _DYNAMIC_CACHE[name]
