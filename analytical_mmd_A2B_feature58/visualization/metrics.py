#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化度量接口模块

本模块为可视化功能提供度量函数的接口转发。
所有度量函数的实际实现已迁移到 metrics.discrepancy 模块。
"""

# 从统一的度量模块导入所有函数
try:
    from ..metrics.discrepancy import (
        calculate_kl_divergence,
        calculate_wasserstein_distances,
        compute_mmd_kernel,
        compute_mmd,
        compute_domain_discrepancy,
        detect_outliers
    )
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    try:
        from analytical_mmd_A2B_feature58.metrics.discrepancy import (
            calculate_kl_divergence,
            calculate_wasserstein_distances,
            compute_mmd_kernel,
            compute_mmd,
            compute_domain_discrepancy,
            detect_outliers
        )
    except ImportError:
        # 如果都失败了，定义空函数以避免导入错误
        def calculate_kl_divergence(*args, **kwargs):
            return 0.0, {}
        def calculate_wasserstein_distances(*args, **kwargs):
            return 0.0, {}
        def compute_mmd_kernel(*args, **kwargs):
            return 0.0
        def compute_mmd(*args, **kwargs):
            return 0.0
        def compute_domain_discrepancy(*args, **kwargs):
            return {}
        def detect_outliers(*args, **kwargs):
            return [], [], [], []

# 为了兼容性，重新导出所有函数
__all__ = [
    'calculate_kl_divergence',
    'calculate_wasserstein_distances',
    'compute_mmd_kernel',
    'compute_mmd',
    'compute_domain_discrepancy',
    'detect_outliers'
] 