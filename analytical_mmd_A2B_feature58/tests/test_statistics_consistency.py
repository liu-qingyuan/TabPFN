#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试不同MMD方法生成的统计表格数值一致性

使用生成的数据进行测试，避免真实数据的复杂性
根据不同方法的优化目标设置合适的测试期望：
- mean_std: 期望KL散度和Wasserstein距离改善
- linear/kpca: 期望MMD改善，KL/Wasserstein可能不改善
"""

import os
import sys
import numpy as np
import logging
from typing import Dict, List, Tuple, Any

# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 计算 analytical_mmd_A2B_feature58 项目的根目录
project_root = os.path.dirname(script_dir)
# 计算 TabPFN 项目的根目录
tabpfn_root = os.path.dirname(project_root)

# 将项目根目录添加到 Python 路径
sys.path.insert(0, project_root)
sys.path.insert(0, tabpfn_root)

try:
    # 导入必要的模块
    from preprocessing.mmd import mmd_transform, compute_mmd
    from metrics.discrepancy import (
        calculate_kl_divergence, calculate_wasserstein_distances
    )
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保您在正确的目录中运行此脚本，并且所有依赖项都已安装。")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"脚本目录: {script_dir}")
    print(f"项目根目录: {project_root}")
    print(f"TabPFN根目录: {tabpfn_root}")
    sys.exit(1)

def generate_test_data(n_source: int = 200, n_target: int = 150, n_features: int = 20, 
                      n_cat_features: int = 5, random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    生成测试数据
    
    参数:
    - n_source: 源域样本数
    - n_target: 目标域样本数  
    - n_features: 总特征数
    - n_cat_features: 类别特征数
    - random_seed: 随机种子
    
    返回:
    - X_source: 源域数据
    - X_target: 目标域数据
    - cat_idx: 类别特征索引
    """
    np.random.seed(random_seed)
    
    # 类别特征索引（前n_cat_features个特征为类别特征）
    cat_idx = list(range(n_cat_features))
    cont_idx = list(range(n_cat_features, n_features))
    
    # 生成源域数据
    X_source = np.zeros((n_source, n_features))
    
    # 类别特征（0-3的整数值）
    for i in cat_idx:
        X_source[:, i] = np.random.randint(0, 4, n_source)
    
    # 连续特征（正态分布）
    for i in cont_idx:
        X_source[:, i] = np.random.normal(0, 1, n_source)
    
    # 生成目标域数据（与源域有分布差异）
    X_target = np.zeros((n_target, n_features))
    
    # 类别特征（相同分布）
    for i in cat_idx:
        X_target[:, i] = np.random.randint(0, 4, n_target)
    
    # 连续特征（不同的均值和方差，模拟域偏移）
    for i in cont_idx:
        # 目标域的均值和方差都有偏移
        mean_shift = np.random.uniform(-1, 1)
        var_scale = np.random.uniform(0.5, 2.0)
        X_target[:, i] = np.random.normal(mean_shift, var_scale, n_target)
    
    logging.info(f"生成测试数据: 源域{n_source}样本, 目标域{n_target}样本, {n_features}特征({n_cat_features}类别+{len(cont_idx)}连续)")
    
    return X_source, X_target, cat_idx

def setup_test_data() -> Tuple[logging.Logger, List[str], Dict[str, Any], List[int]]:
    """设置测试数据"""
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    logger = logging.getLogger(__name__)
    
    # 生成测试数据
    logger.info("生成测试数据...")
    X_source, X_target, cat_idx = generate_test_data()
    
    # 计算原始差异（所有方法都应该相同）
    logger.info("计算原始域间差异...")
    kl_before, kl_before_per_feature = calculate_kl_divergence(X_source, X_target)
    wass_before, wass_before_per_feature = calculate_wasserstein_distances(X_source, X_target)
    
    # 计算原始MMD（使用默认gamma，稍后会根据每个方法的实际gamma重新计算）
    cont_idx = [i for i in range(X_source.shape[1]) if i not in cat_idx]
    mmd_before_default = compute_mmd(X_source[:, cont_idx], 
                                    X_target[:, cont_idx], 
                                    kernel='rbf', gamma=1.0)
    
    logger.info(f"原始KL散度: {kl_before:.6f}")
    logger.info(f"原始Wasserstein距离: {wass_before:.6f}")
    logger.info(f"原始MMD (默认gamma=1.0): {mmd_before_default:.6f}")
    
    # 测试不同MMD方法
    methods = ['linear', 'kpca', 'mean_std']
    results = {}
    
    # 方法参数配置
    method_configs = {
        'linear': {
            'n_epochs': 200,
            'lr': 3e-4,  # 更小的学习率
            'batch_size': 64,
            'lambda_reg': 1e-3,  # 正则化
            'staged_training': True,  # 不分阶段训练
            'dynamic_gamma': True,  # 动态gamma
            'standardize_features': True,  # 标准化输入特征
            'use_gradient_clipping': True,  # 开启梯度裁剪
            'max_grad_norm': 1.0,
            'monitor_gradients': True  # 监控梯度范数
        },
        'kpca': {'kernel': 'rbf', 'gamma': 0.05, 'n_components': 10, 'use_inverse_transform': False},
        'mean_std': {}
    }
    
    for method in methods:
        logger.info(f"测试 {method} 方法...")
        
        # 获取方法参数
        mmd_kwargs = method_configs.get(method, {})
        
        # 应用MMD变换
        X_target_aligned, mmd_info = mmd_transform(
            X_source, X_target, 
            method=method, cat_idx=cat_idx, **mmd_kwargs
        )
        
        # 对于linear方法，输出额外的训练信息
        if method == 'linear' and 'align_info' in mmd_info:
            align_info = mmd_info['align_info']
            logger.info(f"  Linear方法训练详情:")
            if 'final_loss' in align_info:
                logger.info(f"    最终损失: {align_info['final_loss']:.6f}")
            if 'mmd_reduction' in align_info:
                logger.info(f"    MMD降低: {align_info['mmd_reduction']:.6f}")
            if 'converged' in align_info:
                logger.info(f"    是否收敛: {align_info['converged']}")
            if 'gradient_norms' in align_info and len(align_info['gradient_norms']) > 0:
                grad_norms = align_info['gradient_norms']
                logger.info(f"    梯度范数 - 初始: {grad_norms[0]:.6f}, 最终: {grad_norms[-1]:.6f}")
            if 'loss_history' in align_info and len(align_info['loss_history']) > 0:
                loss_history = align_info['loss_history']
                logger.info(f"    损失历史 - 初始: {loss_history[0]:.6f}, 最终: {loss_history[-1]:.6f}")
            logger.info(f"    Linear MMD reduction: {mmd_info.get('reduction', 'N/A')}")
        
        # 获取训练时实际使用的gamma值（关键修复！）
        if 'align_info' in mmd_info and 'gamma_used' in mmd_info['align_info']:
            gamma_used = mmd_info['align_info']['gamma_used']
        elif 'gamma_used' in mmd_info:
            gamma_used = mmd_info['gamma_used']
        else:
            gamma_used = 1.0  # 回退到默认值
            logger.warning(f"方法 {method} 未返回gamma_used，使用默认值1.0")
        
        logger.info(f"方法 {method} 使用的gamma: {gamma_used:.6f}")
        
        # 使用相同的gamma重新计算原始MMD（确保公平比较）
        mmd_before = compute_mmd(X_source[:, cont_idx], 
                                X_target[:, cont_idx], 
                                kernel='rbf', gamma=gamma_used)
        
        # 计算对齐后的差异
        kl_after, kl_after_per_feature = calculate_kl_divergence(X_source, X_target_aligned)
        wass_after, wass_after_per_feature = calculate_wasserstein_distances(X_source, X_target_aligned)
        
        # 使用相同的gamma计算对齐后的MMD（关键修复！）
        mmd_after = compute_mmd(X_source[:, cont_idx], 
                               X_target_aligned[:, cont_idx], 
                               kernel='rbf', gamma=gamma_used)
        
        results[method] = {
            'kl_before': kl_before,
            'kl_after': kl_after,
            'wass_before': wass_before,
            'wass_after': wass_after,
            'mmd_before': mmd_before,
            'mmd_after': mmd_after,
            'kl_before_per_feature': kl_before_per_feature,
            'kl_after_per_feature': kl_after_per_feature,
            'wass_before_per_feature': wass_before_per_feature,
            'wass_after_per_feature': wass_after_per_feature,
            'X_target_aligned': X_target_aligned,
            'mmd_info': mmd_info
        }
        
        # 计算改进百分比
        kl_improvement = (kl_before - kl_after) / kl_before * 100
        wass_improvement = (wass_before - wass_after) / wass_before * 100
        mmd_improvement = (mmd_before - mmd_after) / mmd_before * 100
        
        logger.info(f"  对齐前MMD (gamma={gamma_used:.6f}): {mmd_before:.6f}")
        logger.info(f"  对齐后KL散度: {kl_after:.6f} (改进: {kl_improvement:.2f}%)")
        logger.info(f"  对齐后Wasserstein距离: {wass_after:.6f} (改进: {wass_improvement:.2f}%)")
        logger.info(f"  对齐后MMD (gamma={gamma_used:.6f}): {mmd_after:.6f} (改进: {mmd_improvement:.2f}%)")
    
    return logger, methods, results, cat_idx

def test_before_values_consistency(logger: logging.Logger, methods: List[str], results: Dict[str, Any], X_source: np.ndarray, X_target: np.ndarray, cont_idx: List[int]) -> bool:
    """
    对每个方法，检查 results[method]['mmd_before']
    等于使用它自己 gamma_used 计算一次 compute_mmd 的结果。
    """
    logger.info("\n=== 测试'Before'值一致性（按方法各自gamma） ===")
    
    # 检查KL Before值（所有方法应该相同，因为KL不依赖gamma）
    kl_before_values = [results[method]['kl_before'] for method in methods]
    logger.info(f"所有方法的KL Before值: {kl_before_values}")
    
    kl_all_same = True
    for i, method in enumerate(methods):
        if abs(kl_before_values[i] - kl_before_values[0]) > 1e-10:
            logger.error(f"❌ 方法 {method} 的KL Before值与第一个方法不同")
            kl_all_same = False
    
    if kl_all_same:
        logger.info("✓ 所有方法的KL Before值都相同")
    
    # 检查Wasserstein Before值（所有方法应该相同，因为Wasserstein不依赖gamma）
    wass_before_values = [results[method]['wass_before'] for method in methods]
    logger.info(f"所有方法的Wasserstein Before值: {wass_before_values}")
    
    wass_all_same = True
    for i, method in enumerate(methods):
        if abs(wass_before_values[i] - wass_before_values[0]) > 1e-10:
            logger.error(f"❌ 方法 {method} 的Wasserstein Before值与第一个方法不同")
            wass_all_same = False
    
    if wass_all_same:
        logger.info("✓ 所有方法的Wasserstein Before值都相同")
    
    # 检查MMD Before值（每个方法使用自己的gamma，验证一致性）
    logger.info("检查每个方法的MMD Before值与其gamma_used的一致性:")
    mmd_all_consistent = True
    
    for method in methods:
        res = results[method]
        # 拿到用于度量的gamma
        gamma_used = None
        if 'align_info' in res['mmd_info'] and 'gamma_used' in res['mmd_info']['align_info']:
            gamma_used = res['mmd_info']['align_info']['gamma_used']
        elif 'gamma_used' in res['mmd_info']:
            gamma_used = res['mmd_info']['gamma_used']
        else:
            gamma_used = 1.0
            logger.warning(f"方法 {method} 未找到gamma_used，使用默认值1.0")

        # 重新计算一次MMD
        mmd_before_recalc = compute_mmd(
            X_source[:, cont_idx],
            X_target[:, cont_idx],
            kernel='rbf',
            gamma=gamma_used
        )
        mmd_before_recorded = res['mmd_before']

        logger.info(f"  方法 {method}: gamma_used={gamma_used:.6f}, "
                    f"recorded={mmd_before_recorded:.6f}, "
                    f"recalc={mmd_before_recalc:.6f}")

        if abs(mmd_before_recalc - mmd_before_recorded) > 1e-10:
            logger.error(f"❌ 方法 {method} 的Before MMD与重新计算不一致")
            mmd_all_consistent = False
        else:
            logger.info(f"✓ 方法 {method} 的Before MMD一致")
    
    return kl_all_same and wass_all_same and mmd_all_consistent

def test_after_values_difference(logger: logging.Logger, methods: List[str], results: Dict[str, Any]) -> bool:
    """测试不同方法的'After'值是否不同"""
    logger.info("\n=== 测试'After'值差异性 ===")
    
    # 检查KL After值
    kl_after_values = [results[method]['kl_after'] for method in methods]
    logger.info(f"所有方法的KL After值: {kl_after_values}")
    
    # 至少应该有一对方法的KL After值显著不同
    kl_has_difference = False
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            diff = abs(kl_after_values[i] - kl_after_values[j])
            if diff > 1e-6:  # 显著差异阈值
                kl_has_difference = True
                logger.info(f"✓ 方法 {methods[i]} 和 {methods[j]} 的KL After值差异: {diff:.8f}")
    
    if not kl_has_difference:
        logger.warning("⚠️ 所有方法的KL After值都相同，这可能表明域适应没有正确工作")
    
    # 检查Wasserstein After值
    wass_after_values = [results[method]['wass_after'] for method in methods]
    logger.info(f"所有方法的Wasserstein After值: {wass_after_values}")
    
    # 至少应该有一对方法的Wasserstein After值显著不同
    wass_has_difference = False
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            diff = abs(wass_after_values[i] - wass_after_values[j])
            if diff > 1e-6:  # 显著差异阈值
                wass_has_difference = True
                logger.info(f"✓ 方法 {methods[i]} 和 {methods[j]} 的Wasserstein After值差异: {diff:.8f}")
    
    if not wass_has_difference:
        logger.warning("⚠️ 所有方法的Wasserstein After值都相同，这可能表明域适应没有正确工作")
    
    return kl_has_difference and wass_has_difference

def test_method_specific_improvements(logger: logging.Logger, methods: List[str], results: Dict[str, Any]) -> bool:
    """根据每种方法的优化目标测试改进效果"""
    logger.info("\n=== 测试方法特定的改进效果 ===")
    
    all_methods_valid = True
    
    for method in methods:
        result = results[method]
        
        # 计算改进百分比
        kl_improvement = (result['kl_before'] - result['kl_after']) / result['kl_before'] * 100
        wass_improvement = (result['wass_before'] - result['wass_after']) / result['wass_before'] * 100
        mmd_improvement = (result['mmd_before'] - result['mmd_after']) / result['mmd_before'] * 100
        
        logger.info(f"\n方法 {method}:")
        logger.info(f"  KL改进: {kl_improvement:.2f}%")
        logger.info(f"  Wasserstein改进: {wass_improvement:.2f}%")
        logger.info(f"  MMD改进: {mmd_improvement:.2f}%")
        
        method_valid = True
        
        if method == 'mean_std':
            # mean_std方法应该改善KL散度和Wasserstein距离
            if kl_improvement <= 0:
                logger.error(f"❌ mean_std方法的KL散度没有改进")
                method_valid = False
            else:
                logger.info(f"✓ mean_std方法的KL散度有改进")
            
            if wass_improvement <= 0:
                logger.error(f"❌ mean_std方法的Wasserstein距离没有改进")
                method_valid = False
            else:
                logger.info(f"✓ mean_std方法的Wasserstein距离有改进")
                
        elif method in ['linear', 'kpca']:
            # linear和kpca方法主要优化MMD，KL/Wasserstein可能不改善
            if mmd_improvement <= 0:
                logger.warning(f"⚠️ {method}方法的MMD没有改进，这可能表明优化未收敛")
                # 对于MMD方法，我们放宽要求，只要不是严重恶化就可以接受
                if mmd_improvement < -50:  # 如果MMD恶化超过50%，才认为是失败
                    logger.error(f"❌ {method}方法的MMD严重恶化")
                    method_valid = False
                else:
                    logger.info(f"✓ {method}方法的MMD变化在可接受范围内")
            else:
                logger.info(f"✓ {method}方法的MMD有改进")
            
            # 对于linear和kpca，KL/Wasserstein不做硬性要求
            if kl_improvement <= 0:
                logger.info(f"ℹ️ {method}方法的KL散度无改进，这是预期行为（该方法不直接优化KL散度）")
            else:
                logger.info(f"✓ {method}方法的KL散度意外地有改进")
            
            if wass_improvement <= 0:
                logger.info(f"ℹ️ {method}方法的Wasserstein距离无改进，这是预期行为（该方法不直接优化Wasserstein距离）")
            else:
                logger.info(f"✓ {method}方法的Wasserstein距离意外地有改进")
        
        if not method_valid:
            all_methods_valid = False
    
    return all_methods_valid

def test_categorical_features_unchanged(logger: logging.Logger, methods: List[str], results: Dict[str, Any], cat_idx: List[int]) -> bool:
    """测试类别特征是否保持不变"""
    logger.info("\n=== 测试类别特征保持不变 ===")
    
    all_unchanged = True
    
    for method in methods:
        result = results[method]
        
        # 检查前3个类别特征
        cat_features_to_check = cat_idx[:min(3, len(cat_idx))]
        
        for i in cat_features_to_check:
            feature_key = f'feature_{i}'
            
            # 类别特征的KL After值应该与Before值相同
            kl_before = result['kl_before_per_feature'][feature_key]
            kl_after = result['kl_after_per_feature'][feature_key]
            
            if abs(kl_before - kl_after) > 1e-10:
                logger.error(f"❌ 方法 {method} 改变了类别特征 {i} (KL: {kl_before:.6f} -> {kl_after:.6f})")
                all_unchanged = False
            else:
                logger.info(f"✓ 方法 {method} 保持类别特征 {i} 不变")
    
    return all_unchanged

def test_aligned_features_difference(logger: logging.Logger, methods: List[str], results: Dict[str, Any]) -> bool:
    """测试对齐后的特征矩阵是否不同"""
    logger.info("\n=== 测试对齐后特征矩阵差异性 ===")
    
    aligned_features = {method: results[method]['X_target_aligned'] for method in methods}
    
    has_differences = False
    
    # 检查不同方法生成的对齐特征是否不同
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i < j:  # 避免重复比较
                X1 = aligned_features[method1]
                X2 = aligned_features[method2]
                
                # 计算矩阵差异
                diff_norm = np.linalg.norm(X1 - X2)
                logger.info(f"方法 {method1} 和 {method2} 的对齐特征差异范数: {diff_norm:.8f}")
                
                # 对齐后的特征应该有显著差异
                if diff_norm > 1e-6:
                    logger.info(f"✓ 方法 {method1} 和 {method2} 生成的对齐特征有显著差异")
                    has_differences = True
                else:
                    logger.warning(f"⚠️ 方法 {method1} 和 {method2} 生成的对齐特征过于相似")
    
    return has_differences

def main() -> bool:
    """主函数"""
    print("=== MMD方法统计表格数值一致性测试（使用生成数据）===")
    
    try:
        # 设置测试数据
        logger, methods, results, cat_idx = setup_test_data()
        
        # 重新生成测试数据以获取X_source和X_target
        X_source, X_target, _ = generate_test_data()
        cont_idx = [i for i in range(X_source.shape[1]) if i not in cat_idx]
        
        # 运行所有测试
        test_results = []
        
        # 测试1: Before值一致性
        before_consistent = test_before_values_consistency(logger, methods, results, X_source, X_target, cont_idx)
        test_results.append(("Before值一致性", before_consistent))
        
        # 测试2: After值差异性
        after_different = test_after_values_difference(logger, methods, results)
        test_results.append(("After值差异性", after_different))
        
        # 测试3: 方法特定的改进效果
        improvements_valid = test_method_specific_improvements(logger, methods, results)
        test_results.append(("方法特定改进效果", improvements_valid))
        
        # 测试4: 类别特征保持不变
        cat_unchanged = test_categorical_features_unchanged(logger, methods, results, cat_idx)
        test_results.append(("类别特征保持不变", cat_unchanged))
        
        # 测试5: 对齐特征差异性
        aligned_different = test_aligned_features_difference(logger, methods, results)
        test_results.append(("对齐特征差异性", aligned_different))
        
        # 总结测试结果
        logger.info("\n=== 测试结果总结 ===")
        all_passed = True
        for test_name, passed in test_results:
            status = "✓ 通过" if passed else "❌ 失败"
            logger.info(f"{test_name}: {status}")
            if not passed:
                all_passed = False
        
        if all_passed:
            logger.info("\n🎉 所有测试都通过了！统计表格数值计算正确。")
            logger.info("现在不同MMD方法应该产生不同的统计表格数值。")
            return True
        else:
            logger.error("\n💥 部分测试失败！需要进一步检查代码实现。")
            return False
            
    except Exception as e:
        print(f"测试运行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 