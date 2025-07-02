"""
Pytest configuration for UDA Medical Imbalance Project tests
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest


@pytest.fixture(scope="session")
def project_root_path():
    """返回项目根目录路径"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def sample_config():
    """返回测试用的配置"""
    return {
        'feature_types': ['best7', 'best10', 'all'],
        'datasets': ['A', 'B', 'C'],
        'uda_methods': ['coral', 'linear', 'dann'],
        'imbalance_methods': ['smote', 'borderline', 'adasyn']
    } 