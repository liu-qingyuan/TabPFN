import logging
import sys
import os
from datetime import datetime
from typing import Optional

def setup_logger(log_level: int = logging.INFO, log_file: Optional[str] = None, console_output: bool = True) -> logging.Logger:
    """
    设置日志配置
    
    参数:
    - log_level: 日志级别
    - log_file: 日志文件路径，如果为None则不保存到文件
    - console_output: 是否输出到控制台
    """
    # 清除现有的处理器
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 设置日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 设置根日志器
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # 控制台输出
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件输出
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_experiment_logger(experiment_name: str, base_path: str = "./logs") -> logging.Logger:
    """
    为特定实验创建日志器
    
    参数:
    - experiment_name: 实验名称
    - base_path: 日志文件基础路径
    
    返回:
    - logger: 配置好的日志器
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(base_path, f"{experiment_name}_{timestamp}.log")
    
    return setup_logger(
        log_level=logging.INFO,
        log_file=log_file,
        console_output=True
    ) 