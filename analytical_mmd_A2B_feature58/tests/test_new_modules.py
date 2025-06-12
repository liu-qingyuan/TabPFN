#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试新创建的类条件MMD和阈值优化模块
"""

import numpy as np
import sys
import os

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

def test_class_conditional_mmd():
    """测试类条件MMD模块"""
    print("测试类条件MMD模块...")
    
    try:
        from preprocessing.class_conditional_mmd import (
            generate_pseudo_labels, create_partial_labels, class_conditional_mmd_transform
        )
        print("✓ 类条件MMD模块导入成功")
        
        # 创建有明显分布差异的测试数据
        np.random.seed(42)
        # 源域：均值为0的正态分布
        X_source = np.random.randn(100, 10)
        y_source = np.random.randint(0, 2, 100)
        
        # 目标域：均值为2的正态分布，确保有明显的分布差异
        X_target = np.random.randn(80, 10) + 2.0  # 添加偏移量创建分布差异
        y_target = np.random.randint(0, 2, 80)
        
        # 定义类别特征索引（假设前3个特征是类别特征）
        cat_idx = [0, 1, 2]
        
        # 测试伪标签生成
        pseudo_labels = generate_pseudo_labels(X_source, y_source, X_target)
        print(f"✓ 伪标签生成成功，分布: {np.bincount(pseudo_labels)}")
        
        # 测试部分标签创建
        partial_labels, labeled_indices = create_partial_labels(y_target, label_ratio=0.2)
        print(f"✓ 部分标签创建成功，已标记样本数: {len(labeled_indices)}")
        print(f"  部分标签分布: {np.bincount(partial_labels[partial_labels != -1])}")
        
        # 测试类条件MMD变换（提供cat_idx参数）
        try:
            X_target_transformed, mmd_info = class_conditional_mmd_transform(
                X_source, y_source, X_target, 
                method='mean_std',  # 使用简单的mean_std方法进行测试
                cat_idx=cat_idx     # 提供必需的cat_idx参数
            )
            print(f"✓ 类条件MMD变换测试成功，输出形状: {X_target_transformed.shape}")
            print(f"  整体MMD减少: {mmd_info['overall_reduction']:.2f}%")
            
            # 添加断言检查
            assert X_target_transformed.shape == X_target.shape, "变换后形状应保持不变"
            assert mmd_info['overall_initial_mmd'] > 0, "初始MMD应大于0"
            print(f"  初始MMD: {mmd_info['overall_initial_mmd']:.6f}")
            print(f"  最终MMD: {mmd_info['overall_final_mmd']:.6f}")
            
        except Exception as transform_error:
            print(f"❌ 类条件MMD变换测试失败: {str(transform_error)}")
            return False
        
        print("✓ 类条件MMD模块测试通过")
        
    except Exception as e:
        print(f"❌ 类条件MMD模块测试失败: {str(e)}")
        return False
    
    return True

def test_threshold_optimizer():
    """测试阈值优化模块"""
    print("\n测试阈值优化模块...")
    
    try:
        from preprocessing.threshold_optimizer import (
            optimize_threshold_youden, apply_threshold_optimization, get_roc_curve_data
        )
        print("✓ 阈值优化模块导入成功")
        
        # 创建更有区分度的测试数据
        np.random.seed(42)
        n_samples = 200
        
        # 创建有明显区分度的数据：正类概率偏高，负类概率偏低
        y_true = np.random.randint(0, 2, n_samples)
        y_proba = np.random.rand(n_samples)
        
        # 调整概率使其更有区分度
        positive_mask = (y_true == 1)
        negative_mask = (y_true == 0)
        y_proba[positive_mask] = y_proba[positive_mask] * 0.5 + 0.5  # 正类概率偏向0.5-1.0
        y_proba[negative_mask] = y_proba[negative_mask] * 0.5        # 负类概率偏向0.0-0.5
        
        y_pred = (y_proba > 0.5).astype(int)
        
        # 测试阈值优化
        optimal_threshold, optimal_metrics = optimize_threshold_youden(y_true, y_proba)
        print(f"✓ 阈值优化成功，最佳阈值: {optimal_threshold:.4f}")
        print(f"  Youden指数: {optimal_metrics['youden_index']:.4f}")
        print(f"  最佳阈值下准确率: {optimal_metrics['acc']:.4f}")
        
        # 添加断言检查
        assert 0 <= optimal_threshold <= 1, "最佳阈值应在[0,1]范围内"
        assert 'youden_index' in optimal_metrics, "应包含Youden指数"
        assert 'acc' in optimal_metrics, "应包含准确率"
        
        # 测试应用阈值优化
        y_pred_optimized, optimization_info = apply_threshold_optimization(y_true, y_pred, y_proba)
        print(f"✓ 阈值优化应用成功，准确率改进: {optimization_info['improvements']['acc']:+.4f}")
        
        # 修复：使用正确的字典键名
        print(f"  优化后准确率: {optimization_info['optimal_metrics']['acc']:.4f}")
        print(f"  原始准确率: {optimization_info['original_metrics']['acc']:.4f}")
        
        # 添加断言检查
        assert 'optimal_metrics' in optimization_info, "应包含optimal_metrics"
        assert 'original_metrics' in optimization_info, "应包含original_metrics"
        assert 'improvements' in optimization_info, "应包含improvements"
        assert len(y_pred_optimized) == len(y_true), "优化后预测长度应与真实标签一致"
        
        # 测试ROC曲线数据获取
        roc_data = get_roc_curve_data(y_true, y_proba, optimal_threshold)
        print(f"✓ ROC曲线数据获取成功，数据点数: {len(roc_data['fpr'])}")
        
        # 添加断言检查
        assert len(roc_data['fpr']) == len(roc_data['tpr']), "FPR和TPR长度应一致"
        assert 'optimal_threshold' in roc_data, "应包含最佳阈值信息"
        
        print("✓ 阈值优化模块测试通过")
        
    except Exception as e:
        print(f"❌ 阈值优化模块测试失败: {str(e)}")
        return False
    
    return True

def test_mmd_module():
    """测试原有MMD模块"""
    print("\n测试原有MMD模块...")
    
    try:
        from preprocessing.mmd import mmd_transform, compute_mmd
        print("✓ MMD模块导入成功")
        
        # 创建有明显分布差异的测试数据
        np.random.seed(42)
        # 源域：均值为0，标准差为1
        X_source = np.random.randn(100, 10)
        # 目标域：均值为3，标准差为2，确保有明显的分布差异
        X_target = np.random.randn(80, 10) * 2 + 3
        
        # 定义类别特征索引
        cat_idx = [0, 2, 4]  # 假设的类别特征索引
        
        # 测试MMD计算
        mmd_value = compute_mmd(X_source, X_target)
        print(f"✓ MMD计算成功，MMD值: {mmd_value:.6f}")
        
        # 添加断言检查：确保MMD值大于0（因为我们创建了有差异的分布）
        assert mmd_value > 0, f"MMD值应大于0，实际值: {mmd_value}"
        print(f"  ✓ MMD值验证通过（{mmd_value:.6f} > 0）")
        
        # 测试MMD变换
        X_target_aligned, mmd_info = mmd_transform(X_source, X_target, method='mean_std', cat_idx=cat_idx)
        print(f"✓ MMD变换成功，MMD减少: {mmd_info['reduction']:.2f}%")
        print(f"  变换后形状: {X_target_aligned.shape}")
        print(f"  初始MMD: {mmd_info['initial_mmd']:.6f}")
        print(f"  最终MMD: {mmd_info['final_mmd']:.6f}")
        
        # 添加断言检查
        assert X_target_aligned.shape == X_target.shape, "变换后形状应保持不变"
        assert mmd_info['initial_mmd'] > 0, "初始MMD应大于0"
        assert mmd_info['final_mmd'] >= 0, "最终MMD应非负"
        assert mmd_info['reduction'] >= 0, "MMD减少应为非负值（表示改善）"
        print(f"  ✓ MMD减少验证通过（{mmd_info['reduction']:.2f}% >= 0）")
        
        print("✓ MMD模块测试通过")
        
    except Exception as e:
        print(f"❌ MMD模块测试失败: {str(e)}")
        return False
    
    return True

def main():
    """主测试函数"""
    print("=== 新模块功能测试 ===")
    print("🛠️ 修复说明：")
    print("1. 创建有明显分布差异的测试数据确保MMD > 0")
    print("2. 为类条件MMD提供必需的cat_idx参数")
    print("3. 修复阈值优化的字典键访问问题")
    print("4. 添加断言检查确保测试有效性")
    print()
    
    success_count = 0
    total_tests = 3
    
    # 测试各个模块
    if test_mmd_module():
        success_count += 1
    
    if test_class_conditional_mmd():
        success_count += 1
    
    if test_threshold_optimizer():
        success_count += 1
    
    # 输出测试结果
    print(f"\n=== 测试结果 ===")
    print(f"通过测试: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 所有模块测试通过！")
        return True
    else:
        print("❌ 部分模块测试失败")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 