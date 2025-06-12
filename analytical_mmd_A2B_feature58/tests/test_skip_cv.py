#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试跳过数据集A交叉验证功能的简单脚本
"""

import os
import sys
import logging

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from modeling.cross_domain_runner import run_cross_domain_experiment

def test_skip_cv():
    """测试跳过交叉验证功能"""
    logging.basicConfig(level=logging.INFO)
    
    print("测试1: 正常运行（包含数据集A的交叉验证）")
    try:
        results1 = run_cross_domain_experiment(
            model_type='rf',  # 使用RF模型，速度较快
            feature_type='best7',
            mmd_method='mean_std',  # 使用简单的mean_std方法
            skip_cv_on_a=False,
            save_path='./test_results_with_cv',
            save_visualizations=False
        )
        print(f"✓ 测试1完成，数据集A交叉验证结果: {results1['cross_validation_A'] is not None}")
    except Exception as e:
        print(f"✗ 测试1失败: {e}")
    
    print("\n测试2: 跳过数据集A的交叉验证")
    try:
        results2 = run_cross_domain_experiment(
            model_type='rf',  # 使用RF模型，速度较快
            feature_type='best7',
            mmd_method='mean_std',  # 使用简单的mean_std方法
            skip_cv_on_a=True,
            save_path='./test_results_skip_cv',
            save_visualizations=False
        )
        print(f"✓ 测试2完成，数据集A交叉验证结果: {results2['cross_validation_A'] is not None}")
        print(f"  数据集B外部验证结果存在: {'external_validation_B' in results2}")
    except Exception as e:
        print(f"✗ 测试2失败: {e}")

if __name__ == "__main__":
    test_skip_cv() 