import logging
import sys

def init_logging(level: int = logging.INFO):
    # 先移除 root logger 里所有的 handler
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # 创建两个 handler：
    stdout_handler = logging.StreamHandler(sys.stdout)  # 处理 INFO 及以上的日志
    stderr_handler = logging.StreamHandler(sys.stderr)  # 处理 WARNING 及以上的日志

    # 设置不同的日志级别：
    stdout_handler.setLevel(logging.INFO)   # 只处理 INFO及以上
    stderr_handler.setLevel(logging.WARNING)  # 只处理 WARNING 及以上

    # 设置日志格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stdout_handler.setFormatter(formatter)
    stderr_handler.setFormatter(formatter)

    # 把 handler 添加到 root logger
    logging.root.addHandler(stdout_handler)
    logging.root.addHandler(stderr_handler)
    logging.root.setLevel(level) 