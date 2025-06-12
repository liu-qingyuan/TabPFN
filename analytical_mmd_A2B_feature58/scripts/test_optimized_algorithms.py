#!/usr/bin/env python3
"""
测试优化后的域适应算法

这个脚本测试了以下优化：
1. TCA算法：稀疏特征值求解、可选核矩阵中心化、简洁MMD矩阵构建
2. JDA算法：动态类别权重、置信度阈值伪标签更新、稳健分类器
3. CORAL和Mean-Variance Alignment算法
4. 数值稳定性和性能改进
"""

import sys
import os
import logging
import numpy as np
import time
from typing import Dict, Any, Callable

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.tca import tca_transform, adaptive_tca_transform
from preprocessing.jda import jda_transform, adaptive_jda_transform
from preprocessing.coral import coral_transform, adaptive_coral_transform
from preprocessing.mean_variance_alignment import mean_variance_transform, adaptive_mean_variance_transform
from data.data_loader import load_data_with_features
from config.settings import BEST_7_FEATURES, BEST_7_CAT_IDX

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimized_algorithms_test.log'),
        logging.StreamHandler()
    ]
)

def test_algorithm_performance(algorithm_name: str, transform_func: Callable, X_s: np.ndarray, y_s: np.ndarray, X_t: np.ndarray, **params: Any) -> Dict[str, Any]:
    """测试单个算法的性能"""
    logging.info(f"\n测试{algorithm_name}...")
    
    start_time = time.time()
    try:
        if 'jda' in algorithm_name.lower():
            result = transform_func(X_s, y_s, X_t, **params)
        else:
            result = transform_func(X_s, X_t, **params)
        
        # 处理不同的返回值格式
        if len(result) == 3:
            # TCA和JDA方法：返回(X_s_trans, X_t_trans, info)
            X_s_trans, X_t_trans, info = result
        elif len(result) == 2:
            # CORAL和Mean-Variance方法：返回(X_t_trans, info)
            X_t_trans, info = result
            X_s_trans = X_s  # 源域保持不变
        else:
            raise ValueError(f"意外的返回值数量: {len(result)}")
        
        elapsed_time = time.time() - start_time
        
        # 检查数值稳定性
        has_nan = np.any(np.isnan(X_s_trans)) or np.any(np.isnan(X_t_trans))
        
        logging.info(f"✓ {algorithm_name}成功")
        logging.info(f"  执行时间: {elapsed_time:.3f}秒")
        logging.info(f"  变换后源域形状: {X_s_trans.shape}")
        logging.info(f"  变换后目标域形状: {X_t_trans.shape}")
        logging.info(f"  数值稳定性: {'❌ 有NaN' if has_nan else '✅ 稳定'}")
        
        if isinstance(info, dict) and 'improvement_percent' in info:
            logging.info(f"  改进百分比: {info['improvement_percent']:.2f}%")
        
        return {
            'success': True,
            'time': elapsed_time,
            'has_nan': has_nan,
            'info': info
        }
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logging.error(f"✗ {algorithm_name}失败: {e}")
        import traceback
        logging.error(f"详细错误信息: {traceback.format_exc()}")
        return {
            'success': False,
            'time': elapsed_time,
            'error': str(e)
        }


def test_optimized_tca():
    """测试优化后的TCA算法"""
    logging.info("=" * 60)
    logging.info("测试优化后的TCA域适应算法")
    logging.info("=" * 60)
    
    # 加载数据
    X_s, y_s = load_data_with_features('A', BEST_7_FEATURES, BEST_7_CAT_IDX)
    X_t, _ = load_data_with_features('B', BEST_7_FEATURES, BEST_7_CAT_IDX)
    
    logging.info(f"源域A: {X_s.shape[0]} 样本, {X_s.shape[1]} 特征")
    logging.info(f"目标域B: {X_t.shape[0]} 样本, {X_t.shape[1]} 特征")
    
    # 测试不同配置
    test_configs = [
        {
            'name': 'TCA（密集求解器，中心化核）',
            'func': tca_transform,
            'params': {'subspace_dim': 5, 'center_kernel': True, 'use_sparse_solver': False}
        },
        {
            'name': 'TCA（稀疏求解器，中心化核）',
            'func': tca_transform,
            'params': {'subspace_dim': 5, 'center_kernel': True, 'use_sparse_solver': True}
        },
        {
            'name': 'TCA（稀疏求解器，非中心化核）',
            'func': tca_transform,
            'params': {'subspace_dim': 5, 'center_kernel': False, 'use_sparse_solver': True}
        },
        {
            'name': 'Adaptive TCA',
            'func': adaptive_tca_transform,
            'params': {'subspace_dim_range': (3, 10), 'n_trials': 5}
        }
    ]
    
    results = []
    for config in test_configs:
        result = test_algorithm_performance(config['name'], config['func'], 
                                          X_s, y_s, X_t, **config['params'])
        results.append(result)
    
    return results


def test_optimized_jda():
    """测试优化后的JDA算法"""
    logging.info("=" * 60)
    logging.info("测试优化后的JDA域适应算法")
    logging.info("=" * 60)
    
    # 加载数据
    X_s, y_s = load_data_with_features('A', BEST_7_FEATURES, BEST_7_CAT_IDX)
    X_t, _ = load_data_with_features('B', BEST_7_FEATURES, BEST_7_CAT_IDX)
    
    logging.info(f"源域A: {X_s.shape[0]} 样本, {X_s.shape[1]} 特征")
    logging.info(f"目标域B: {X_t.shape[0]} 样本, {X_t.shape[1]} 特征")
    
    # 测试不同配置
    test_configs = [
        {
            'name': 'JDA（基本配置）',
            'func': jda_transform,
            'params': {
                'subspace_dim': 5, 'max_iterations': 3, 'mu': 0.5,
                'center_kernel': True, 'use_sparse_solver': False, 'confidence_threshold': 0.7
            }
        },
        {
            'name': 'JDA（稀疏求解器）',
            'func': jda_transform,
            'params': {
                'subspace_dim': 5, 'max_iterations': 3, 'mu': 0.5,
                'center_kernel': True, 'use_sparse_solver': True, 'confidence_threshold': 0.7
            }
        },
        {
            'name': 'JDA（高置信度阈值）',
            'func': jda_transform,
            'params': {
                'subspace_dim': 5, 'max_iterations': 3, 'mu': 0.5,
                'center_kernel': False, 'use_sparse_solver': True, 'confidence_threshold': 0.8
            }
        },
        {
            'name': 'Adaptive JDA',
            'func': adaptive_jda_transform,
            'params': {
                'subspace_dim_range': (3, 10), 'mu_range': (0.3, 0.7), 
                'n_trials': 5, 'max_iterations': 2
            }
        }
    ]
    
    results = []
    for config in test_configs:
        result = test_algorithm_performance(config['name'], config['func'], 
                                          X_s, y_s, X_t, **config['params'])
        results.append(result)
    
    return results


def test_coral_and_mean_variance():
    """测试CORAL和Mean-Variance Alignment算法"""
    logging.info("=" * 60)
    logging.info("测试CORAL和Mean-Variance Alignment算法")
    logging.info("=" * 60)
    
    # 加载数据
    X_s, y_s = load_data_with_features('A', BEST_7_FEATURES, BEST_7_CAT_IDX)
    X_t, _ = load_data_with_features('B', BEST_7_FEATURES, BEST_7_CAT_IDX)
    
    logging.info(f"源域A: {X_s.shape[0]} 样本, {X_s.shape[1]} 特征")
    logging.info(f"目标域B: {X_t.shape[0]} 样本, {X_t.shape[1]} 特征")
    
    # 测试配置
    test_configs = [
        {
            'name': 'CORAL（基本）',
            'func': coral_transform,
            'params': {'regularization': 1e-6}
        },
        {
            'name': 'Adaptive CORAL',
            'func': adaptive_coral_transform,
            'params': {'regularization_range': (1e-8, 1e-3), 'n_trials': 5}
        },
        {
            'name': 'Mean-Variance Alignment',
            'func': mean_variance_transform,
            'params': {'align_mean': True, 'align_variance': True}
        },
        {
            'name': 'Adaptive Mean-Variance',
            'func': adaptive_mean_variance_transform,
            'params': {}
        }
    ]
    
    results = []
    for config in test_configs:
        result = test_algorithm_performance(config['name'], config['func'], 
                                          X_s, y_s, X_t, **config['params'])
        results.append(result)
    
    return results


def test_all_adaptive_methods():
    """测试所有自适应域适应方法"""
    logging.info("=" * 60)
    logging.info("测试所有自适应域适应方法")
    logging.info("=" * 60)
    
    # 加载数据
    X_s, y_s = load_data_with_features('A', BEST_7_FEATURES, BEST_7_CAT_IDX)
    X_t, _ = load_data_with_features('B', BEST_7_FEATURES, BEST_7_CAT_IDX)
    
    logging.info(f"源域A: {X_s.shape[0]} 样本, {X_s.shape[1]} 特征")
    logging.info(f"目标域B: {X_t.shape[0]} 样本, {X_t.shape[1]} 特征")
    
    # 所有自适应方法配置
    adaptive_configs = [
        {
            'name': 'Adaptive TCA（快速版）',
            'func': adaptive_tca_transform,
            'params': {'subspace_dim_range': (3, 8), 'n_trials': 3}
        },
        {
            'name': 'Adaptive JDA（快速版）',
            'func': adaptive_jda_transform,
            'params': {
                'subspace_dim_range': (3, 8), 'mu_range': (0.3, 0.7), 
                'n_trials': 3, 'max_iterations': 2
            }
        },
        {
            'name': 'Adaptive CORAL（快速版）',
            'func': adaptive_coral_transform,
            'params': {'regularization_range': (1e-8, 1e-3), 'n_trials': 3}
        },
        {
            'name': 'Adaptive Mean-Variance',
            'func': adaptive_mean_variance_transform,
            'params': {}
        }
    ]
    
    results = []
    for config in adaptive_configs:
        result = test_algorithm_performance(config['name'], config['func'], 
                                          X_s, y_s, X_t, **config['params'])
        results.append(result)
    
    return results


def test_parameter_sensitivity():
    """测试参数敏感性"""
    logging.info("=" * 60)
    logging.info("参数敏感性测试")
    logging.info("=" * 60)
    
    # 加载数据
    X_s, y_s = load_data_with_features('A', BEST_7_FEATURES, BEST_7_CAT_IDX)
    X_t, _ = load_data_with_features('B', BEST_7_FEATURES, BEST_7_CAT_IDX)
    
    # 测试TCA的子空间维度敏感性
    logging.info("1. TCA子空间维度敏感性测试...")
    subspace_dims = [3, 5, 7]
    for dim in subspace_dims:
        result = test_algorithm_performance(
            f'TCA(dim={dim})', tca_transform, X_s, y_s, X_t,
            subspace_dim=dim, use_sparse_solver=True, center_kernel=False
        )
        if result['success']:
            improvement = result['info'].get('improvement_percent', 0)
            logging.info(f"  维度{dim}: 改进{improvement:.2f}%, 时间{result['time']:.3f}s")
    
    # 测试JDA的mu参数敏感性
    logging.info("\n2. JDA权重参数mu敏感性测试...")
    mu_values = [0.3, 0.5, 0.7]
    for mu in mu_values:
        result = test_algorithm_performance(
            f'JDA(mu={mu})', jda_transform, X_s, y_s, X_t,
            subspace_dim=5, mu=mu, max_iterations=2, 
            use_sparse_solver=True, confidence_threshold=0.7
        )
        if result['success']:
            improvement = result['info'].get('improvement_percent', 0)
            logging.info(f"  mu={mu}: 改进{improvement:.2f}%, 时间{result['time']:.3f}s")


def main():
    """主函数"""
    logging.info("开始测试所有域适应算法...")
    
    try:
        # 测试优化后的TCA（包括自适应版本）
        logging.info("🔄 开始TCA算法测试...")
        tca_results = test_optimized_tca()
        
        # 测试优化后的JDA（包括自适应版本）
        logging.info("🔄 开始JDA算法测试...")
        jda_results = test_optimized_jda()
        
        # 测试CORAL和Mean-Variance（包括自适应版本）
        logging.info("🔄 开始CORAL和Mean-Variance算法测试...")
        coral_mv_results = test_coral_and_mean_variance()
        
        # 测试所有自适应方法的综合对比
        logging.info("🔄 开始自适应方法综合测试...")
        adaptive_results = test_all_adaptive_methods()
        
        # 参数敏感性测试
        logging.info("🔄 开始参数敏感性测试...")
        test_parameter_sensitivity()
        
        # 生成总结
        logging.info("=" * 80)
        logging.info("🎯 所有域适应算法测试总结")
        logging.info("=" * 80)
        
        all_results = tca_results + jda_results + coral_mv_results + adaptive_results
        successful = sum(1 for r in all_results if r['success'])
        total = len(all_results)
        
        logging.info(f"📊 测试统计:")
        logging.info(f"  总测试数: {total}")
        logging.info(f"  成功测试: {successful}")
        logging.info(f"  成功率: {successful/total*100:.1f}%")
        
        if successful > 0:
            avg_time = np.mean([r['time'] for r in all_results if r['success']])
            stable_count = sum(1 for r in all_results if r['success'] and not r['has_nan'])
            logging.info(f"  平均执行时间: {avg_time:.3f}秒")
            logging.info(f"  数值稳定率: {stable_count}/{successful} ({stable_count/successful*100:.1f}%)")
        
        # 按类别统计结果
        logging.info(f"\n📈 分类别结果:")
        logging.info(f"  TCA方法: {sum(1 for r in tca_results if r['success'])}/{len(tca_results)} 成功")
        logging.info(f"  JDA方法: {sum(1 for r in jda_results if r['success'])}/{len(jda_results)} 成功")
        logging.info(f"  CORAL/Mean-Variance方法: {sum(1 for r in coral_mv_results if r['success'])}/{len(coral_mv_results)} 成功")
        logging.info(f"  自适应方法: {sum(1 for r in adaptive_results if r['success'])}/{len(adaptive_results)} 成功")
        
        # 性能最佳的方法
        if successful > 0:
            best_methods = []
            for result in all_results:
                if result['success'] and 'improvement_percent' in result['info']:
                    improvement = result['info']['improvement_percent']
                    best_methods.append((result, improvement))
            
            if best_methods:
                best_methods.sort(key=lambda x: x[1], reverse=True)
                logging.info(f"\n🏆 性能最佳的前3个方法:")
                for i, (result, improvement) in enumerate(best_methods[:3]):
                    method_name = "未知方法"
                    for r in all_results:
                        if r == result:
                            # 从日志中提取方法名称（这里简化处理）
                            method_name = f"方法{i+1}"
                            break
                    logging.info(f"  {i+1}. 改进: {improvement:.2f}%, 时间: {result['time']:.3f}s")
        
        logging.info("=" * 80)
        logging.info("✅ 所有域适应算法测试完成!")
        logging.info("=" * 80)
        
    except Exception as e:
        logging.error(f"❌ 测试过程中发生错误: {e}")
        import traceback
        logging.error(f"详细错误信息: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main() 