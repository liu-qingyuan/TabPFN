"""
UDA Medical Imbalance Project - 异常处理模块

定义项目中使用的自定义异常类，提供更精确的错误信息和处理。

作者: UDA Medical Team  
日期: 2024-01-30
"""

from typing import Optional, Any, Dict
import traceback
import logging

logger = logging.getLogger(__name__)

class UDAMedicalError(Exception):
    """
    UDA医疗项目基础异常类
    
    所有项目自定义异常的基类，提供统一的错误处理接口。
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """
        初始化异常
        
        Args:
            message: 错误信息
            error_code: 错误代码
            details: 错误详细信息字典
            cause: 原始异常（如果有）
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.cause = cause
        
        # 记录异常
        self._log_exception()
    
    def _log_exception(self):
        """记录异常信息"""
        log_message = f"[{self.error_code}] {self.message}"
        if self.details:
            log_message += f" - Details: {self.details}"
        if self.cause:
            log_message += f" - Caused by: {self.cause}"
        
        logger.error(log_message)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
            'cause': str(self.cause) if self.cause else None,
            'traceback': traceback.format_exc()
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        result = f"[{self.error_code}] {self.message}"
        if self.details:
            result += f" (Details: {self.details})"
        return result


class DataValidationError(UDAMedicalError):
    """
    数据验证异常
    
    当数据格式、内容或结构不符合要求时抛出。
    """
    
    def __init__(
        self, 
        message: str,
        data_info: Optional[Dict[str, Any]] = None,
        expected_format: Optional[str] = None,
        actual_format: Optional[str] = None,
        **kwargs
    ):
        details = {
            'data_info': data_info,
            'expected_format': expected_format,
            'actual_format': actual_format
        }
        super().__init__(
            message, 
            error_code="DATA_VALIDATION_ERROR",
            details=details,
            **kwargs
        )


class ModelConfigurationError(UDAMedicalError):
    """
    模型配置异常
    
    当模型参数配置错误或不兼容时抛出。
    """
    
    def __init__(
        self, 
        message: str,
        model_type: Optional[str] = None,
        invalid_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        details = {
            'model_type': model_type,
            'invalid_params': invalid_params
        }
        super().__init__(
            message,
            error_code="MODEL_CONFIG_ERROR", 
            details=details,
            **kwargs
        )


class UDAMethodError(UDAMedicalError):
    """
    UDA方法异常
    
    当UDA方法执行失败或配置错误时抛出。
    """
    
    def __init__(
        self, 
        message: str,
        method_name: Optional[str] = None,
        method_params: Optional[Dict[str, Any]] = None,
        execution_stage: Optional[str] = None,
        **kwargs
    ):
        details = {
            'method_name': method_name,
            'method_params': method_params,
            'execution_stage': execution_stage
        }
        super().__init__(
            message,
            error_code="UDA_METHOD_ERROR",
            details=details,
            **kwargs
        )


class PreprocessingError(UDAMedicalError):
    """
    数据预处理异常
    
    当数据预处理（标准化、特征选择、不平衡处理等）失败时抛出。
    """
    
    def __init__(
        self, 
        message: str,
        preprocessing_step: Optional[str] = None,
        input_shape: Optional[tuple] = None,
        output_shape: Optional[tuple] = None,
        **kwargs
    ):
        details = {
            'preprocessing_step': preprocessing_step,
            'input_shape': input_shape,
            'output_shape': output_shape
        }
        super().__init__(
            message,
            error_code="PREPROCESSING_ERROR",
            details=details,
            **kwargs
        )


class EvaluationError(UDAMedicalError):
    """
    模型评估异常
    
    当模型评估过程中出现错误时抛出。
    """
    
    def __init__(
        self, 
        message: str,
        metric_name: Optional[str] = None,
        evaluation_stage: Optional[str] = None,
        data_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        details = {
            'metric_name': metric_name,
            'evaluation_stage': evaluation_stage,
            'data_info': data_info
        }
        super().__init__(
            message,
            error_code="EVALUATION_ERROR",
            details=details,
            **kwargs
        )


class FeatureSelectionError(UDAMedicalError):
    """
    特征选择异常
    
    当特征选择过程中出现错误时抛出。
    """
    
    def __init__(
        self, 
        message: str,
        feature_method: Optional[str] = None,
        requested_features: Optional[int] = None,
        available_features: Optional[int] = None,
        **kwargs
    ):
        details = {
            'feature_method': feature_method,
            'requested_features': requested_features,
            'available_features': available_features
        }
        super().__init__(
            message,
            error_code="FEATURE_SELECTION_ERROR",
            details=details,
            **kwargs
        )


class ResourceError(UDAMedicalError):
    """
    资源异常
    
    当内存、GPU或其他计算资源不足时抛出。
    """
    
    def __init__(
        self, 
        message: str,
        resource_type: Optional[str] = None,
        required_amount: Optional[str] = None,
        available_amount: Optional[str] = None,
        **kwargs
    ):
        details = {
            'resource_type': resource_type,
            'required_amount': required_amount,
            'available_amount': available_amount
        }
        super().__init__(
            message,
            error_code="RESOURCE_ERROR",
            details=details,
            **kwargs
        )


class FileOperationError(UDAMedicalError):
    """
    文件操作异常
    
    当文件读写操作失败时抛出。
    """
    
    def __init__(
        self, 
        message: str,
        file_path: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        details = {
            'file_path': file_path,
            'operation': operation
        }
        super().__init__(
            message,
            error_code="FILE_OPERATION_ERROR",
            details=details,
            **kwargs
        )


# 异常处理装饰器
def handle_exceptions(
    default_return=None,
    reraise: bool = True,
    log_error: bool = True
):
    """
    异常处理装饰器
    
    Args:
        default_return: 发生异常时的默认返回值
        reraise: 是否重新抛出异常
        log_error: 是否记录错误日志
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except UDAMedicalError:
                # UDA医疗项目的自定义异常直接传递
                if reraise:
                    raise
                return default_return
            except Exception as e:
                # 其他异常包装为UDAMedicalError
                if log_error:
                    logger.error(f"Unexpected error in {func.__name__}: {e}")
                
                wrapped_error = UDAMedicalError(
                    f"Unexpected error in {func.__name__}: {str(e)}",
                    error_code="UNEXPECTED_ERROR",
                    details={'function': func.__name__, 'args': str(args)[:200]},
                    cause=e
                )
                
                if reraise:
                    raise wrapped_error
                return default_return
        
        return wrapper
    return decorator


# 异常上下文管理器
class ExceptionContext:
    """异常上下文管理器"""
    
    def __init__(
        self, 
        operation_name: str,
        suppress_exceptions: bool = False,
        default_return=None
    ):
        self.operation_name = operation_name
        self.suppress_exceptions = suppress_exceptions
        self.default_return = default_return
        self.exception_occurred = False
        self.exception_info = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.exception_occurred = True
            self.exception_info = {
                'type': exc_type.__name__,
                'value': str(exc_val),
                'operation': self.operation_name
            }
            
            logger.error(f"Exception in {self.operation_name}: {exc_val}")
            
            if self.suppress_exceptions:
                return True  # 抑制异常
        
        return False  # 不抑制异常


# 便捷函数
def safe_execute(func, *args, default=None, **kwargs):
    """
    安全执行函数，捕获所有异常
    
    Args:
        func: 要执行的函数
        *args: 位置参数
        default: 异常时的默认返回值
        **kwargs: 关键字参数
    
    Returns:
        函数执行结果或默认值
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Safe execution failed for {func.__name__}: {e}")
        return default


def validate_and_raise(
    condition: bool,
    exception_class: type = UDAMedicalError,
    message: str = "Validation failed",
    **exception_kwargs
):
    """
    验证条件并在失败时抛出异常
    
    Args:
        condition: 验证条件
        exception_class: 异常类
        message: 异常信息
        **exception_kwargs: 异常参数
    """
    if not condition:
        raise exception_class(message, **exception_kwargs)


# 导出的异常类和函数
__all__ = [
    'UDAMedicalError',
    'DataValidationError',
    'ModelConfigurationError', 
    'UDAMethodError',
    'PreprocessingError',
    'EvaluationError',
    'FeatureSelectionError',
    'ResourceError',
    'FileOperationError',
    'handle_exceptions',
    'ExceptionContext',
    'safe_execute',
    'validate_and_raise'
]