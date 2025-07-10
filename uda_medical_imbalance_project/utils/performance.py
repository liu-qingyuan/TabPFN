"""
UDA Medical Imbalance Project - 性能监控模块

提供性能监控、计时和内存使用跟踪功能，帮助优化代码性能。

作者: UDA Medical Team
日期: 2024-01-30
"""

import time
import psutil
import functools
import logging
from typing import Dict, Any, Optional, Callable, Union
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
import threading
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    
    # 时间指标
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration: Optional[float] = None
    
    # 内存指标
    memory_start: Optional[float] = None
    memory_end: Optional[float] = None
    memory_peak: Optional[float] = None
    memory_used: Optional[float] = None
    
    # CPU指标
    cpu_start: Optional[float] = None
    cpu_end: Optional[float] = None
    cpu_avg: Optional[float] = None
    
    # 系统指标
    process_id: int = field(default_factory=lambda: psutil.Process().pid)
    thread_count: Optional[int] = None
    
    # 自定义指标
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def finalize(self):
        """完成性能测量并计算最终指标"""
        self.end_time = time.time()
        if self.start_time:
            self.duration = self.end_time - self.start_time
        
        # 计算内存使用
        if self.memory_start and self.memory_end:
            self.memory_used = self.memory_end - self.memory_start
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'duration_seconds': self.duration,
            'memory_start_mb': self.memory_start,
            'memory_end_mb': self.memory_end,
            'memory_peak_mb': self.memory_peak,
            'memory_used_mb': self.memory_used,
            'cpu_start_percent': self.cpu_start,
            'cpu_end_percent': self.cpu_end,
            'cpu_avg_percent': self.cpu_avg,
            'process_id': self.process_id,
            'thread_count': self.thread_count,
            'custom_metrics': self.custom_metrics,
            'timestamp': datetime.fromtimestamp(self.start_time).isoformat()
        }


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(
        self,
        name: str,
        track_memory: bool = True,
        track_cpu: bool = True,
        sampling_interval: float = 1.0,
        auto_log: bool = True
    ):
        """
        初始化性能监控器
        
        Args:
            name: 监控器名称
            track_memory: 是否监控内存
            track_cpu: 是否监控CPU
            sampling_interval: 采样间隔（秒）
            auto_log: 是否自动记录日志
        """
        self.name = name
        self.track_memory = track_memory
        self.track_cpu = track_cpu
        self.sampling_interval = sampling_interval
        self.auto_log = auto_log
        
        self.metrics = PerformanceMetrics()
        self.process = psutil.Process()
        
        # 监控状态
        self.is_running = False
        self.monitoring_thread = None
        
        # 采样数据
        self.memory_samples = []
        self.cpu_samples = []
    
    def start(self):
        """开始性能监控"""
        if self.is_running:
            logger.warning(f"性能监控器 {self.name} 已经在运行")
            return
        
        self.metrics = PerformanceMetrics()
        self.is_running = True
        
        # 记录初始状态
        if self.track_memory:
            self.metrics.memory_start = self._get_memory_usage()
        
        if self.track_cpu:
            self.metrics.cpu_start = self._get_cpu_usage()
        
        self.metrics.thread_count = self.process.num_threads()
        
        # 开始后台监控线程
        if self.track_memory or self.track_cpu:
            self.monitoring_thread = threading.Thread(
                target=self._background_monitoring,
                daemon=True
            )
            self.monitoring_thread.start()
        
        if self.auto_log:
            logger.info(f"开始性能监控: {self.name}")
    
    def stop(self) -> PerformanceMetrics:
        """停止性能监控并返回结果"""
        if not self.is_running:
            logger.warning(f"性能监控器 {self.name} 未在运行")
            return self.metrics
        
        self.is_running = False
        
        # 等待监控线程结束
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)
        
        # 记录最终状态
        if self.track_memory:
            self.metrics.memory_end = self._get_memory_usage()
            if self.memory_samples:
                self.metrics.memory_peak = max(self.memory_samples)
        
        if self.track_cpu:
            self.metrics.cpu_end = self._get_cpu_usage()
            if self.cpu_samples:
                self.metrics.cpu_avg = sum(self.cpu_samples) / len(self.cpu_samples)
        
        # 完成性能测量
        self.metrics.finalize()
        
        if self.auto_log:
            self._log_results()
        
        return self.metrics
    
    def add_custom_metric(self, name: str, value: Any):
        """添加自定义指标"""
        self.metrics.custom_metrics[name] = value
    
    def _get_memory_usage(self) -> float:
        """获取当前内存使用量（MB）"""
        try:
            return self.process.memory_info().rss / 1024 / 1024
        except Exception as e:
            logger.warning(f"获取内存使用量失败: {e}")
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """获取当前CPU使用率（%）"""
        try:
            return self.process.cpu_percent()
        except Exception as e:
            logger.warning(f"获取CPU使用率失败: {e}")
            return 0.0
    
    def _background_monitoring(self):
        """后台监控线程"""
        while self.is_running:
            try:
                if self.track_memory:
                    memory_usage = self._get_memory_usage()
                    self.memory_samples.append(memory_usage)
                
                if self.track_cpu:
                    cpu_usage = self._get_cpu_usage()
                    self.cpu_samples.append(cpu_usage)
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                logger.error(f"后台监控线程出错: {e}")
                break
    
    def _log_results(self):
        """记录性能结果"""
        metrics_dict = self.metrics.to_dict()
        
        log_message = f"性能监控结果 [{self.name}]:\n"
        log_message += f"  执行时间: {self.metrics.duration:.2f}秒\n"
        
        if self.track_memory:
            log_message += f"  内存使用: {self.metrics.memory_used:.1f}MB "
            log_message += f"(峰值: {self.metrics.memory_peak:.1f}MB)\n"
        
        if self.track_cpu:
            log_message += f"  CPU使用: 平均{self.metrics.cpu_avg:.1f}%\n"
        
        if self.metrics.custom_metrics:
            log_message += f"  自定义指标: {self.metrics.custom_metrics}\n"
        
        logger.info(log_message.rstrip())
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()


class TimerContext:
    """简单的计时上下文管理器"""
    
    def __init__(self, name: str, auto_log: bool = True):
        self.name = name
        self.auto_log = auto_log
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.time()
        if self.auto_log:
            logger.debug(f"开始计时: {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        
        if self.auto_log:
            logger.info(f"计时结束 [{self.name}]: {self.duration:.3f}秒")


def profile_function(
    track_memory: bool = True,
    track_cpu: bool = False,
    auto_log: bool = True
):
    """
    函数性能分析装饰器
    
    Args:
        track_memory: 是否监控内存
        track_cpu: 是否监控CPU
        auto_log: 是否自动记录日志
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            monitor_name = f"{func.__module__}.{func.__name__}"
            
            with PerformanceMonitor(
                name=monitor_name,
                track_memory=track_memory,
                track_cpu=track_cpu,
                auto_log=auto_log
            ) as monitor:
                
                # 记录函数参数信息
                if args or kwargs:
                    args_info = f"args_count: {len(args)}, kwargs_count: {len(kwargs)}"
                    monitor.add_custom_metric("function_args", args_info)
                
                # 执行函数
                try:
                    result = func(*args, **kwargs)
                    monitor.add_custom_metric("execution_status", "success")
                    return result
                except Exception as e:
                    monitor.add_custom_metric("execution_status", "error")
                    monitor.add_custom_metric("error_type", type(e).__name__)
                    raise
        
        return wrapper
    return decorator


def time_function(func: Callable) -> Callable:
    """简单的函数计时装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with TimerContext(f"{func.__name__}", auto_log=True):
            return func(*args, **kwargs)
    return wrapper


@contextmanager
def monitor_performance(
    name: str,
    track_memory: bool = True,
    track_cpu: bool = False,
    save_to_file: Optional[Path] = None
):
    """
    性能监控上下文管理器
    
    Args:
        name: 监控名称
        track_memory: 是否监控内存
        track_cpu: 是否监控CPU
        save_to_file: 是否保存结果到文件
    """
    monitor = PerformanceMonitor(
        name=name,
        track_memory=track_memory,
        track_cpu=track_cpu,
        auto_log=True
    )
    
    try:
        monitor.start()
        yield monitor
    finally:
        metrics = monitor.stop()
        
        # 保存到文件
        if save_to_file:
            try:
                save_to_file.parent.mkdir(parents=True, exist_ok=True)
                with open(save_to_file, 'w', encoding='utf-8') as f:
                    json.dump(metrics.to_dict(), f, indent=2, ensure_ascii=False)
                logger.info(f"性能指标已保存到: {save_to_file}")
            except Exception as e:
                logger.error(f"保存性能指标失败: {e}")


def memory_usage() -> Dict[str, float]:
    """
    获取当前内存使用情况
    
    Returns:
        包含内存使用信息的字典
    """
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # 物理内存
            'vms_mb': memory_info.vms / 1024 / 1024,  # 虚拟内存
            'percent': process.memory_percent(),       # 内存使用百分比
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    except Exception as e:
        logger.error(f"获取内存使用情况失败: {e}")
        return {}


def system_resources() -> Dict[str, Any]:
    """
    获取系统资源使用情况
    
    Returns:
        包含系统资源信息的字典
    """
    try:
        # CPU信息
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 内存信息
        memory = psutil.virtual_memory()
        
        # 磁盘信息
        disk = psutil.disk_usage('/')
        
        return {
            'cpu': {
                'count': cpu_count,
                'usage_percent': cpu_percent,
            },
            'memory': {
                'total_gb': memory.total / 1024 / 1024 / 1024,
                'available_gb': memory.available / 1024 / 1024 / 1024,
                'used_percent': memory.percent
            },
            'disk': {
                'total_gb': disk.total / 1024 / 1024 / 1024,
                'free_gb': disk.free / 1024 / 1024 / 1024,
                'used_percent': (disk.used / disk.total) * 100
            }
        }
    except Exception as e:
        logger.error(f"获取系统资源信息失败: {e}")
        return {}


class PerformanceTracker:
    """性能跟踪器 - 跟踪多个操作的性能"""
    
    def __init__(self, name: str = "PerformanceTracker"):
        self.name = name
        self.operations = {}
        self.global_start_time = time.time()
    
    def start_operation(self, operation_name: str) -> TimerContext:
        """开始跟踪一个操作"""
        timer = TimerContext(operation_name, auto_log=False)
        self.operations[operation_name] = timer
        timer.__enter__()
        return timer
    
    def end_operation(self, operation_name: str):
        """结束跟踪一个操作"""
        if operation_name in self.operations:
            timer = self.operations[operation_name]
            timer.__exit__(None, None, None)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        total_time = time.time() - self.global_start_time
        
        operation_times = {}
        for name, timer in self.operations.items():
            if timer.duration is not None:
                operation_times[name] = timer.duration
        
        return {
            'tracker_name': self.name,
            'total_time': total_time,
            'operation_times': operation_times,
            'operation_count': len(operation_times),
            'memory_usage': memory_usage(),
            'timestamp': datetime.now().isoformat()
        }
    
    def log_summary(self):
        """记录性能摘要日志"""
        summary = self.get_summary()
        
        log_message = f"性能跟踪摘要 [{self.name}]:\n"
        log_message += f"  总时间: {summary['total_time']:.2f}秒\n"
        log_message += f"  操作数量: {summary['operation_count']}\n"
        
        for op_name, duration in summary['operation_times'].items():
            log_message += f"  - {op_name}: {duration:.3f}秒\n"
        
        logger.info(log_message.rstrip())


# 导出的类和函数
__all__ = [
    'PerformanceMetrics',
    'PerformanceMonitor',
    'TimerContext',
    'PerformanceTracker',
    'profile_function',
    'time_function',
    'monitor_performance',
    'memory_usage',
    'system_resources'
]