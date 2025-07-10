"""
UDA Medical Imbalance Project - 通用工具函数模块

提供数据转换、文件操作、格式化和其他通用辅助功能。

作者: UDA Medical Team
日期: 2024-01-30
"""

import numpy as np
import pandas as pd
import json
import pickle
import logging
from typing import Union, Any, Dict, List, Optional, Tuple
from pathlib import Path
import shutil
import hashlib
from datetime import datetime, timedelta
import re
import warnings

from .exceptions import FileOperationError, DataValidationError

logger = logging.getLogger(__name__)

# 数据转换工具
def ensure_array(
    data: Union[np.ndarray, pd.DataFrame, pd.Series, list],
    dtype: Optional[type] = None
) -> np.ndarray:
    """
    确保数据是numpy数组格式
    
    Args:
        data: 输入数据
        dtype: 目标数据类型
        
    Returns:
        numpy数组
    """
    try:
        if isinstance(data, np.ndarray):
            result = data
        elif isinstance(data, pd.DataFrame):
            result = data.values
        elif isinstance(data, pd.Series):
            result = data.values
        elif isinstance(data, list):
            result = np.array(data)
        else:
            result = np.array(data)
        
        if dtype is not None:
            result = result.astype(dtype)
        
        return result
        
    except Exception as e:
        raise DataValidationError(
            f"无法转换数据为numpy数组: {str(e)}",
            data_info={'input_type': type(data).__name__},
            cause=e
        )


def ensure_dataframe(
    data: Union[np.ndarray, pd.DataFrame, dict],
    columns: Optional[List[str]] = None,
    index: Optional[Union[List, pd.Index]] = None
) -> pd.DataFrame:
    """
    确保数据是pandas DataFrame格式
    
    Args:
        data: 输入数据
        columns: 列名列表
        index: 索引
        
    Returns:
        pandas DataFrame
    """
    try:
        if isinstance(data, pd.DataFrame):
            result = data.copy()
        elif isinstance(data, np.ndarray):
            if columns is None and len(data.shape) == 2:
                columns = [f'feature_{i}' for i in range(data.shape[1])]
            result = pd.DataFrame(data, columns=columns, index=index)
        elif isinstance(data, dict):
            result = pd.DataFrame(data, index=index)
        else:
            result = pd.DataFrame(data, columns=columns, index=index)
        
        return result
        
    except Exception as e:
        raise DataValidationError(
            f"无法转换数据为DataFrame: {str(e)}",
            data_info={'input_type': type(data).__name__},
            cause=e
        )


def safe_divide(
    numerator: Union[float, int, np.ndarray],
    denominator: Union[float, int, np.ndarray],
    default_value: float = 0.0
) -> Union[float, np.ndarray]:
    """
    安全除法，避免除零错误
    
    Args:
        numerator: 分子
        denominator: 分母
        default_value: 除零时的默认值
        
    Returns:
        除法结果
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            
            if isinstance(denominator, (int, float)):
                if denominator == 0:
                    return default_value
                return numerator / denominator
            else:
                # 数组情况
                denominator = np.asarray(denominator)
                numerator = np.asarray(numerator)
                
                result = np.where(
                    denominator != 0,
                    numerator / denominator,
                    default_value
                )
                return result
                
    except Exception as e:
        logger.warning(f"除法运算失败: {e}")
        return default_value


# 时间和格式化工具
def format_duration(seconds: float) -> str:
    """
    格式化时间长度为可读字符串
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串
    """
    if seconds < 1:
        return f"{seconds*1000:.1f}毫秒"
    elif seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}小时"


def format_memory_size(bytes_size: Union[int, float]) -> str:
    """
    格式化内存大小为可读字符串
    
    Args:
        bytes_size: 字节数
        
    Returns:
        格式化的内存大小字符串
    """
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(bytes_size)
    
    for unit in units:
        if size < 1024.0:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    
    return f"{size:.1f}PB"


def format_percentage(value: float, precision: int = 1) -> str:
    """
    格式化百分比
    
    Args:
        value: 数值（0-1或0-100）
        precision: 小数位数
        
    Returns:
        格式化的百分比字符串
    """
    if value <= 1.0:
        percentage = value * 100
    else:
        percentage = value
    
    return f"{percentage:.{precision}f}%"


# 文件操作工具
def setup_directories(*dirs: Union[str, Path]) -> List[Path]:
    """
    创建目录（如果不存在）
    
    Args:
        *dirs: 目录路径列表
        
    Returns:
        创建的目录路径列表
    """
    created_dirs = []
    
    for dir_path in dirs:
        try:
            path = Path(dir_path)
            path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(path)
            logger.debug(f"目录已创建或已存在: {path}")
        except Exception as e:
            raise FileOperationError(
                f"创建目录失败: {dir_path}",
                file_path=str(dir_path),
                operation="mkdir",
                cause=e
            )
    
    return created_dirs


def save_json(
    data: Dict[str, Any],
    filepath: Union[str, Path],
    indent: int = 2,
    ensure_ascii: bool = False
):
    """
    保存数据为JSON文件
    
    Args:
        data: 要保存的数据
        filepath: 文件路径
        indent: 缩进空格数
        ensure_ascii: 是否确保ASCII编码
    """
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, default=str)
        
        logger.debug(f"JSON文件已保存: {filepath}")
        
    except Exception as e:
        raise FileOperationError(
            f"保存JSON文件失败: {filepath}",
            file_path=str(filepath),
            operation="save_json",
            cause=e
        )


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    加载JSON文件
    
    Args:
        filepath: 文件路径
        
    Returns:
        加载的数据
    """
    try:
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileOperationError(
                f"JSON文件不存在: {filepath}",
                file_path=str(filepath),
                operation="load_json"
            )
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.debug(f"JSON文件已加载: {filepath}")
        return data
        
    except FileOperationError:
        raise
    except Exception as e:
        raise FileOperationError(
            f"加载JSON文件失败: {filepath}",
            file_path=str(filepath),
            operation="load_json",
            cause=e
        )


def save_pickle(
    data: Any,
    filepath: Union[str, Path],
    protocol: int = pickle.HIGHEST_PROTOCOL
):
    """
    保存数据为pickle文件
    
    Args:
        data: 要保存的数据
        filepath: 文件路径
        protocol: pickle协议版本
    """
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=protocol)
        
        logger.debug(f"Pickle文件已保存: {filepath}")
        
    except Exception as e:
        raise FileOperationError(
            f"保存Pickle文件失败: {filepath}",
            file_path=str(filepath),
            operation="save_pickle",
            cause=e
        )


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    加载pickle文件
    
    Args:
        filepath: 文件路径
        
    Returns:
        加载的数据
    """
    try:
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileOperationError(
                f"Pickle文件不存在: {filepath}",
                file_path=str(filepath),
                operation="load_pickle"
            )
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        logger.debug(f"Pickle文件已加载: {filepath}")
        return data
        
    except FileOperationError:
        raise
    except Exception as e:
        raise FileOperationError(
            f"加载Pickle文件失败: {filepath}",
            file_path=str(filepath),
            operation="load_pickle",
            cause=e
        )


def backup_file(
    filepath: Union[str, Path],
    backup_dir: Optional[Union[str, Path]] = None,
    add_timestamp: bool = True
) -> Path:
    """
    备份文件
    
    Args:
        filepath: 原文件路径
        backup_dir: 备份目录，默认为原文件同目录
        add_timestamp: 是否在备份文件名中添加时间戳
        
    Returns:
        备份文件路径
    """
    try:
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileOperationError(
                f"要备份的文件不存在: {filepath}",
                file_path=str(filepath),
                operation="backup"
            )
        
        # 确定备份目录
        if backup_dir is None:
            backup_dir = filepath.parent / "backups"
        else:
            backup_dir = Path(backup_dir)
        
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成备份文件名
        if add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{filepath.stem}_{timestamp}{filepath.suffix}"
        else:
            backup_name = f"{filepath.stem}_backup{filepath.suffix}"
        
        backup_path = backup_dir / backup_name
        
        # 复制文件
        shutil.copy2(filepath, backup_path)
        
        logger.info(f"文件已备份: {filepath} -> {backup_path}")
        return backup_path
        
    except FileOperationError:
        raise
    except Exception as e:
        raise FileOperationError(
            f"备份文件失败: {filepath}",
            file_path=str(filepath),
            operation="backup",
            cause=e
        )


# 字符串和验证工具
def sanitize_filename(filename: str) -> str:
    """
    清理文件名，移除不安全字符
    
    Args:
        filename: 原文件名
        
    Returns:
        清理后的文件名
    """
    # 移除或替换不安全字符
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # 移除控制字符
    filename = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', filename)
    
    # 限制长度
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        max_name_len = 255 - len(ext) - 1 if ext else 255
        filename = name[:max_name_len] + ('.' + ext if ext else '')
    
    return filename


def calculate_file_hash(
    filepath: Union[str, Path],
    algorithm: str = 'md5'
) -> str:
    """
    计算文件哈希值
    
    Args:
        filepath: 文件路径
        algorithm: 哈希算法 ('md5', 'sha1', 'sha256')
        
    Returns:
        文件哈希值
    """
    try:
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileOperationError(
                f"文件不存在: {filepath}",
                file_path=str(filepath),
                operation="hash"
            )
        
        hash_obj = hashlib.new(algorithm)
        
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
        
    except FileOperationError:
        raise
    except Exception as e:
        raise FileOperationError(
            f"计算文件哈希失败: {filepath}",
            file_path=str(filepath),
            operation="hash",
            cause=e
        )


# 数据处理工具
def split_data_by_ratio(
    data: Union[np.ndarray, pd.DataFrame],
    ratios: List[float],
    random_state: Optional[int] = None
) -> List[Union[np.ndarray, pd.DataFrame]]:
    """
    按比例分割数据
    
    Args:
        data: 要分割的数据
        ratios: 分割比例列表，和应为1.0
        random_state: 随机种子
        
    Returns:
        分割后的数据列表
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"分割比例的和必须为1.0，当前和为: {sum(ratios)}")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(data)
    indices = np.random.permutation(n_samples)
    
    splits = []
    start_idx = 0
    
    for i, ratio in enumerate(ratios):
        if i == len(ratios) - 1:  # 最后一个分割
            end_idx = n_samples
        else:
            end_idx = start_idx + int(n_samples * ratio)
        
        split_indices = indices[start_idx:end_idx]
        
        if isinstance(data, pd.DataFrame):
            splits.append(data.iloc[split_indices])
        else:
            splits.append(data[split_indices])
        
        start_idx = end_idx
    
    return splits


def merge_dictionaries(*dicts: Dict[str, Any], deep_merge: bool = False) -> Dict[str, Any]:
    """
    合并多个字典
    
    Args:
        *dicts: 要合并的字典
        deep_merge: 是否深度合并
        
    Returns:
        合并后的字典
    """
    result = {}
    
    for d in dicts:
        if not isinstance(d, dict):
            continue
            
        for key, value in d.items():
            if key in result and deep_merge:
                if isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dictionaries(result[key], value, deep_merge=True)
                else:
                    result[key] = value
            else:
                result[key] = value
    
    return result


def flatten_dict(
    d: Dict[str, Any],
    separator: str = '.',
    prefix: str = ''
) -> Dict[str, Any]:
    """
    展平嵌套字典
    
    Args:
        d: 嵌套字典
        separator: 键分隔符
        prefix: 键前缀
        
    Returns:
        展平后的字典
    """
    result = {}
    
    for key, value in d.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key
        
        if isinstance(value, dict):
            result.update(flatten_dict(value, separator, new_key))
        else:
            result[new_key] = value
    
    return result


# 类型检查和转换工具
def is_numeric(value: Any) -> bool:
    """检查值是否为数值类型"""
    return isinstance(value, (int, float, np.number))


def is_string_like(value: Any) -> bool:
    """检查值是否为字符串类型"""
    return isinstance(value, (str, bytes))


def to_numeric(value: Any, default: Optional[float] = None) -> Optional[float]:
    """
    尝试将值转换为数值
    
    Args:
        value: 要转换的值
        default: 转换失败时的默认值
        
    Returns:
        转换后的数值或默认值
    """
    try:
        if is_numeric(value):
            return float(value)
        elif is_string_like(value):
            return float(value)
        else:
            return default
    except (ValueError, TypeError):
        return default


# 版本比较工具
def compare_versions(version1: str, version2: str) -> int:
    """
    比较版本号
    
    Args:
        version1: 版本1
        version2: 版本2
        
    Returns:
        -1: version1 < version2
         0: version1 == version2
         1: version1 > version2
    """
    def parse_version(v):
        return [int(x) for x in v.split('.')]
    
    v1_parts = parse_version(version1)
    v2_parts = parse_version(version2)
    
    # 补齐长度
    max_len = max(len(v1_parts), len(v2_parts))
    v1_parts.extend([0] * (max_len - len(v1_parts)))
    v2_parts.extend([0] * (max_len - len(v2_parts)))
    
    for v1, v2 in zip(v1_parts, v2_parts):
        if v1 < v2:
            return -1
        elif v1 > v2:
            return 1
    
    return 0


# 导出的函数
__all__ = [
    # 数据转换
    'ensure_array',
    'ensure_dataframe',
    'safe_divide',
    
    # 格式化
    'format_duration',
    'format_memory_size',
    'format_percentage',
    
    # 文件操作
    'setup_directories',
    'save_json',
    'load_json',
    'save_pickle',
    'load_pickle',
    'backup_file',
    
    # 字符串工具
    'sanitize_filename',
    'calculate_file_hash',
    
    # 数据处理
    'split_data_by_ratio',
    'merge_dictionaries',
    'flatten_dict',
    
    # 类型检查
    'is_numeric',
    'is_string_like',
    'to_numeric',
    
    # 版本比较
    'compare_versions'
]