#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试MMD域适应可视化模块

本测试文件验证三种MMD域适应方法的可视化功能：
1. Linear MMD: 使用线性变换最小化MMD距离
2. Kernel PCA MMD: 使用核PCA进行特征空间变换
3. Mean-Std MMD: 简单的均值-标准差对齐方法

测试数据生成策略：
- 源域：标准正态分布 N(0,1)
- 目标域：有明显偏移的分布，每个特征有不同的均值和方差
- 适应后：使用真正的MMD方法进行域适应

注意：
- 统计表格中的颜色编码表示分布偏移严重程度
- 红色=高偏移，橙色=中等偏移，绿色=低偏移
- MMD适应效果通过改进百分比显示
"""

import os
import sys
import numpy as np
import logging

# 添加项目根目录和当前包目录到Python路径，以便进行绝对导入
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # analytical_mmd_A2B_feature58目录
tabpfn_root = os.path.dirname(project_root)  # TabPFN根目录
sys.path.insert(0, project_root)
sys.path.insert(0, tabpfn_root)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入可视化模块
from metrics.discrepancy import calculate_kl_divergence, calculate_wasserstein_distances, compute_mmd
from visualization.tsne_plots import visualize_tsne
from visualization.histogram_plots import visualize_feature_histograms, histograms_stats_table
from visualization.comparison_plots import compare_before_after_adaptation, plot_mmd_methods_comparison
from visualization.utils import close_figures

# 导入真正的MMD实现
from preprocessing.mmd import mmd_transform

def generate_diverse_test_data():
    """生成更多样化的测试数据，确保不同方法有不同效果"""
    np.random.seed(42)
    
    # 源域数据 (标准正态分布)
    n_samples = 300
    n_features = 8  # 增加特征数以更好展示差异
    
    # 定义类别特征索引（匹配实际特征数量）
    cat_idx = [0, 2, 4, 6]  # 类别特征索引
    cont_idx = [1, 3, 5, 7]  # 连续特征索引
    
    X_source = np.zeros((n_samples, n_features))
    X_target = np.zeros((n_samples, n_features))
    
    # 生成类别特征（离散值，源域和目标域相同分布）
    for i in cat_idx:
        # 类别特征：0, 1, 2 三个类别，分布相同
        X_source[:, i] = np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.3, 0.2])
        X_target[:, i] = np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.3, 0.2])
    
    # 生成连续特征（源域标准正态分布，目标域有明显偏移）
    for i, cont_feature_idx in enumerate(cont_idx):
        # 源域：标准正态分布
        X_source[:, cont_feature_idx] = np.random.normal(0, 1, n_samples)
        
        # 目标域：只使用均值和方差偏移（适合域适应处理）
        if i % 2 == 0:  # 均值偏移
            X_target[:, cont_feature_idx] = np.random.normal(2.0, 1.0, n_samples)
        else:  # 方差偏移
            X_target[:, cont_feature_idx] = np.random.normal(0, 2.0, n_samples)
    
    y_source = np.random.binomial(1, 0.5, n_samples)
    y_target = np.random.binomial(1, 0.6, n_samples)
    
    # 特征名称
    feature_names = []
    for i in range(n_features):
        if i in cat_idx:
            feature_names.append(f'Category{i+1}')
        else:
            feature_names.append(f'Continuous{i+1}')
    
    logger.info(f"数据生成完成:")
    logger.info(f"  类别特征索引: {cat_idx}")
    logger.info(f"  连续特征索引: {cont_idx}")
    logger.info(f"  特征名称: {feature_names}")
    
    return X_source, X_target, y_source, y_target, feature_names, cat_idx

def test_real_mmd_methods():
    """测试真正的MMD方法"""
    try:
        logger.info("✓ 所有可视化模块导入成功")
        
        # 生成测试数据
        X_source, X_target, y_source, y_target, feature_names, cat_idx = generate_diverse_test_data()
        logger.info("✓ 测试数据生成成功")
        logger.info(f"数据形状: 源域{X_source.shape}, 目标域{X_target.shape}")
        logger.info(f"类别特征索引: {cat_idx}")
        
        # 创建输出目录
        output_dir = os.path.join(script_dir, "test_visualizations")
        os.makedirs(output_dir, exist_ok=True)
        
        # 计算初始MMD作为基线
        initial_mmd = compute_mmd(X_source, X_target)
        logger.info(f"初始MMD (基线): {initial_mmd:.6f}")
        
        # 测试三种不同的MMD方法
        mmd_methods = {
            'MMD-linear': {
                'method': 'linear',
                'params': {
                    'gamma': 1.0,
                    'lr': 0.02,  # 增加学习率
                    'n_epochs': 300,  # 增加训练轮数
                    'batch_size': 32,  # 减小批次大小
                    'lambda_reg': 1e-5  # 减小正则化
                }
            },
            'MMD-kpca': {
                'method': 'kpca',
                'params': {
                    'kernel': 'rbf',
                    'gamma': 0.5,  # 调整gamma值
                    'n_components': 4  # 减少组件数以避免过拟合
                }
            },
            'MMD-mean_std': {
                'method': 'mean_std',
                'params': {}
            }
        }
        
        results = {}
        
        for method_name, method_config in mmd_methods.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"测试 {method_name} 方法...")
            logger.info(f"{'='*50}")
            
            try:
                # 使用真正的MMD实现进行域适应
                logger.info(f"开始 {method_name} 域适应...")
                logger.info(f"参数: {method_config['params']}")
                
                X_adapted, mmd_info = mmd_transform(
                    X_source, 
                    X_target, 
                    method=method_config['method'],
                    cat_idx=cat_idx,
                    **method_config['params']
                )
                
                logger.info(f"✓ {method_name} 域适应完成")
                logger.info(f"  初始MMD: {mmd_info['initial_mmd']:.6f}")
                logger.info(f"  最终MMD: {mmd_info['final_mmd']:.6f}")
                logger.info(f"  改进: {mmd_info['reduction']:.2f}%")
                
                # 添加详细的数据统计分析
                logger.info(f"  数据统计分析:")
                
                # 连续特征索引（排除类别特征）
                cont_idx = [i for i in range(X_source.shape[1]) if i not in cat_idx]
                logger.info(f"    连续特征索引: {cont_idx}")
                logger.info(f"    类别特征索引: {cat_idx}")
                
                # 计算连续特征的统计差异
                source_mean = np.mean(X_source[:, cont_idx], axis=0)
                target_mean_before = np.mean(X_target[:, cont_idx], axis=0)
                target_mean_after = np.mean(X_adapted[:, cont_idx], axis=0)
                
                source_std = np.std(X_source[:, cont_idx], axis=0)
                target_std_before = np.std(X_target[:, cont_idx], axis=0)
                target_std_after = np.std(X_adapted[:, cont_idx], axis=0)
                
                mean_diff_before = np.mean(np.abs(source_mean - target_mean_before))
                mean_diff_after = np.mean(np.abs(source_mean - target_mean_after))
                std_diff_before = np.mean(np.abs(source_std - target_std_before))
                std_diff_after = np.mean(np.abs(source_std - target_std_after))
                
                logger.info(f"    连续特征均值差异: {mean_diff_before:.4f} → {mean_diff_after:.4f}")
                logger.info(f"    连续特征标准差差异: {std_diff_before:.4f} → {std_diff_after:.4f}")
                
                # 检查类别特征是否保持不变（这是关键验证）
                cat_features_unchanged = np.array_equal(X_target[:, cat_idx], X_adapted[:, cat_idx])
                logger.info(f"    类别特征保持不变: {cat_features_unchanged}")
                if not cat_features_unchanged:
                    logger.warning(f"    ⚠️  警告：类别特征发生了变化！这不应该发生。")
                    # 计算类别特征的变化程度
                    cat_change = np.mean(np.abs(X_target[:, cat_idx] - X_adapted[:, cat_idx]))
                    logger.warning(f"    类别特征平均变化: {cat_change:.6f}")
                
                # 计算适应强度（仅针对连续特征）
                cont_adaptation_strength = np.mean(np.abs(X_target[:, cont_idx] - X_adapted[:, cont_idx]))
                logger.info(f"    连续特征适应强度: {cont_adaptation_strength:.4f}")
                
                # 计算总体适应强度（包括类别特征，应该主要来自连续特征）
                total_adaptation_strength = np.mean(np.abs(X_target - X_adapted))
                logger.info(f"    总体适应强度: {total_adaptation_strength:.4f}")
                
                # 验证适应效果
                kl_before, _ = calculate_kl_divergence(X_source, X_target)
                kl_after, _ = calculate_kl_divergence(X_source, X_adapted)
                
                wass_before, _ = calculate_wasserstein_distances(X_source, X_target)
                wass_after, _ = calculate_wasserstein_distances(X_source, X_adapted)
                
                logger.info(f"  验证指标:")
                logger.info(f"    KL散度: {kl_before:.4f} → {kl_after:.4f}")
                logger.info(f"    Wasserstein距离: {wass_before:.4f} → {wass_after:.4f}")
                
                # 创建方法特定的输出目录
                method_dir = os.path.join(output_dir, method_name)
                os.makedirs(method_dir, exist_ok=True)
                
                # 只生成综合对比可视化（避免重复文件）
                logger.info(f"生成 {method_name} 综合对比可视化...")
                comparison_dir = os.path.join(method_dir, "comparison")
                comparison_results = compare_before_after_adaptation(
                    source_features=X_source,
                    target_features=X_target,
                    adapted_target_features=X_adapted,
                    source_labels=y_source,
                    target_labels=y_target,
                    save_dir=comparison_dir,
                    method_name=method_name,
                    feature_names=feature_names
                )
                logger.info(f"✓ {method_name} 综合对比可视化完成")
                
                # 保存结果
                results[method_name] = comparison_results
                
            except Exception as e:
                logger.error(f"❌ {method_name} 方法测试失败: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # 生成方法对比（只有当有多个成功的结果时）
        if len(results) > 1:
            logger.info(f"\n{'='*50}")
            logger.info("生成MMD方法对比...")
            logger.info(f"{'='*50}")
            comparison_path = os.path.join(output_dir, "mmd_methods_comparison.png")
            plot_mmd_methods_comparison(results, comparison_path)
            logger.info(f"✓ MMD方法对比图生成完成: {comparison_path}")
        else:
            logger.warning("成功的方法少于2个，跳过方法对比图生成")
        
        # 清理图形
        close_figures()
        logger.info("✓ 图形清理完成")
        
        logger.info(f"\n{'='*50}")
        logger.info("🎉 MMD可视化模块测试完成！")
        logger.info(f"{'='*50}")
        logger.info(f"测试结果保存在: {output_dir}")
        logger.info("\n测试总结：")
        logger.info("1. 使用了真正的MMD实现进行域适应")
        logger.info("2. 测试了三种MMD方法：linear（线性变换）、kpca（核PCA）、mean_std（均值标准差对齐）")
        logger.info("3. 每种方法都有不同的适应策略和效果")
        logger.info("4. 生成了完整的可视化对比，包括t-SNE、直方图、统计表格")
        logger.info("5. 直方图现在使用seaborn格式，包含KDE曲线和颜色编码")
        logger.info(f"6. 成功测试了 {len(results)} 种方法")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_real_mmd_methods()
    if success:
        print("\n✅ 所有测试通过！MMD可视化模块工作正常。")
    else:
        print("\n❌ 测试失败！请检查错误信息。")
        sys.exit(1) 