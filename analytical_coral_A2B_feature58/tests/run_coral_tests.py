#!/usr/bin/env python3
"""
CORAL功能测试运行脚本
用于快速验证CORAL和条件CORAL的基本功能
"""

import sys
import os
import numpy as np
import logging
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.coral import coral_transform, class_conditional_coral_transform, generate_pseudo_labels_for_coral

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_test_data():
    """创建测试数据"""
    logging.info("创建测试数据...")
    
    # 设置随机种子
    np.random.seed(42)
    
    # 生成源域数据
    X_source, y_source = make_classification(
        n_samples=200, n_features=8, n_informative=6, n_redundant=2,
        n_classes=2, random_state=42, class_sep=1.5
    )
    
    # 生成目标域数据（有域偏移）
    X_target, y_target = make_classification(
        n_samples=150, n_features=8, n_informative=6, n_redundant=2,
        n_classes=2, random_state=123, class_sep=1.2
    )
    
    # 添加域偏移
    X_target = X_target * 1.8 + 1.5
    
    # 定义类别特征索引（最后2个特征）
    cat_idx = [6, 7]
    
    # 将类别特征转换为整数
    X_source[:, cat_idx] = np.round(np.abs(X_source[:, cat_idx])).astype(int) % 3
    X_target[:, cat_idx] = np.round(np.abs(X_target[:, cat_idx])).astype(int) % 3
    
    # 标准化
    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_source)
    X_target_scaled = scaler.transform(X_target)
    
    logging.info(f"数据创建完成:")
    logging.info(f"源域: {X_source_scaled.shape}, 标签分布: {np.bincount(y_source)}")
    logging.info(f"目标域: {X_target_scaled.shape}, 标签分布: {np.bincount(y_target)}")
    logging.info(f"类别特征索引: {cat_idx}")
    
    return X_source_scaled, y_source, X_target_scaled, y_target, cat_idx

def test_basic_coral(X_source, X_target, cat_idx):
    """测试基本CORAL功能"""
    logging.info("\n=== 测试基本CORAL功能 ===")
    
    # 计算变换前的域差异
    cont_idx = [i for i in range(X_source.shape[1]) if i not in cat_idx]
    
    mean_diff_before = np.mean(np.abs(
        np.mean(X_source[:, cont_idx], axis=0) - 
        np.mean(X_target[:, cont_idx], axis=0)
    ))
    
    std_diff_before = np.mean(np.abs(
        np.std(X_source[:, cont_idx], axis=0) - 
        np.std(X_target[:, cont_idx], axis=0)
    ))
    
    logging.info(f"变换前 - 均值差异: {mean_diff_before:.6f}, 标准差差异: {std_diff_before:.6f}")
    
    # 执行CORAL变换
    try:
        X_target_aligned = coral_transform(X_source, X_target, cat_idx)
        
        # 计算变换后的域差异
        mean_diff_after = np.mean(np.abs(
            np.mean(X_source[:, cont_idx], axis=0) - 
            np.mean(X_target_aligned[:, cont_idx], axis=0)
        ))
        
        std_diff_after = np.mean(np.abs(
            np.std(X_source[:, cont_idx], axis=0) - 
            np.std(X_target_aligned[:, cont_idx], axis=0)
        ))
        
        logging.info(f"变换后 - 均值差异: {mean_diff_after:.6f}, 标准差差异: {std_diff_after:.6f}")
        
        # 验证类别特征是否保持不变
        if np.array_equal(X_target[:, cat_idx], X_target_aligned[:, cat_idx]):
            logging.info("✓ 类别特征保持不变")
        else:
            logging.error("✗ 类别特征被意外改变")
            return False
        
        # 验证域差异是否减少
        if mean_diff_after < mean_diff_before and std_diff_after < std_diff_before:
            logging.info("✓ CORAL成功减少了域差异")
            improvement_mean = (mean_diff_before - mean_diff_after) / mean_diff_before * 100
            improvement_std = (std_diff_before - std_diff_after) / std_diff_before * 100
            logging.info(f"改善程度 - 均值: {improvement_mean:.1f}%, 标准差: {improvement_std:.1f}%")
            return True
        else:
            logging.warning("⚠ CORAL未能有效减少域差异")
            return False
            
    except Exception as e:
        logging.error(f"✗ CORAL变换失败: {e}")
        return False

def test_class_conditional_coral(X_source, y_source, X_target, y_target, cat_idx):
    """测试类条件CORAL功能"""
    logging.info("\n=== 测试类条件CORAL功能 ===")
    
    try:
        # 生成伪标签
        logging.info("生成伪标签...")
        yt_pseudo = generate_pseudo_labels_for_coral(X_source, y_source, X_target, cat_idx)
        
        logging.info(f"伪标签分布: {np.bincount(yt_pseudo)}")
        logging.info(f"真实标签分布: {np.bincount(y_target)}")
        
        # 计算伪标签准确率
        pseudo_accuracy = np.mean(yt_pseudo == y_target)
        logging.info(f"伪标签准确率: {pseudo_accuracy:.3f}")
        
        # 执行类条件CORAL变换
        logging.info("执行类条件CORAL变换...")
        X_target_aligned = class_conditional_coral_transform(
            X_source, y_source, X_target, yt_pseudo, cat_idx, alpha=0.1
        )
        
        # 验证输出形状
        if X_target_aligned.shape == X_target.shape:
            logging.info("✓ 输出形状正确")
        else:
            logging.error("✗ 输出形状不正确")
            return False
        
        # 验证类别特征是否保持不变
        if np.array_equal(X_target[:, cat_idx], X_target_aligned[:, cat_idx]):
            logging.info("✓ 类别特征保持不变")
        else:
            logging.error("✗ 类别特征被意外改变")
            return False
        
        # 计算域差异改善
        cont_idx = [i for i in range(X_source.shape[1]) if i not in cat_idx]
        
        mean_diff_before = np.mean(np.abs(
            np.mean(X_source[:, cont_idx], axis=0) - 
            np.mean(X_target[:, cont_idx], axis=0)
        ))
        
        mean_diff_after = np.mean(np.abs(
            np.mean(X_source[:, cont_idx], axis=0) - 
            np.mean(X_target_aligned[:, cont_idx], axis=0)
        ))
        
        if mean_diff_after < mean_diff_before:
            improvement = (mean_diff_before - mean_diff_after) / mean_diff_before * 100
            logging.info(f"✓ 类条件CORAL减少了域差异: {improvement:.1f}%")
            return True
        else:
            logging.warning("⚠ 类条件CORAL未能有效减少域差异")
            return False
            
    except Exception as e:
        logging.error(f"✗ 类条件CORAL变换失败: {e}")
        return False

def compare_coral_methods(X_source, y_source, X_target, y_target, cat_idx):
    """比较不同CORAL方法的效果"""
    logging.info("\n=== 比较CORAL方法效果 ===")
    
    cont_idx = [i for i in range(X_source.shape[1]) if i not in cat_idx]
    
    # 原始域差异
    original_diff = np.mean(np.abs(
        np.mean(X_source[:, cont_idx], axis=0) - 
        np.mean(X_target[:, cont_idx], axis=0)
    ))
    
    try:
        # 普通CORAL
        X_target_coral = coral_transform(X_source, X_target, cat_idx)
        coral_diff = np.mean(np.abs(
            np.mean(X_source[:, cont_idx], axis=0) - 
            np.mean(X_target_coral[:, cont_idx], axis=0)
        ))
        
        # 类条件CORAL
        yt_pseudo = generate_pseudo_labels_for_coral(X_source, y_source, X_target, cat_idx)
        X_target_class_coral = class_conditional_coral_transform(
            X_source, y_source, X_target, yt_pseudo, cat_idx, alpha=0.1
        )
        class_coral_diff = np.mean(np.abs(
            np.mean(X_source[:, cont_idx], axis=0) - 
            np.mean(X_target_class_coral[:, cont_idx], axis=0)
        ))
        
        # 输出比较结果
        logging.info(f"原始域差异: {original_diff:.6f}")
        logging.info(f"普通CORAL后: {coral_diff:.6f} (改善 {(1-coral_diff/original_diff)*100:.1f}%)")
        logging.info(f"类条件CORAL后: {class_coral_diff:.6f} (改善 {(1-class_coral_diff/original_diff)*100:.1f}%)")
        
        if coral_diff < original_diff and class_coral_diff < original_diff:
            logging.info("✓ 两种CORAL方法都有效减少了域差异")
            return True
        else:
            logging.warning("⚠ 部分CORAL方法效果不佳")
            return False
            
    except Exception as e:
        logging.error(f"✗ CORAL方法比较失败: {e}")
        return False

def main():
    """主测试函数"""
    logging.info("开始CORAL功能测试...")
    
    # 创建测试数据
    X_source, y_source, X_target, y_target, cat_idx = create_test_data()
    
    # 测试结果
    results = []
    
    # 测试基本CORAL
    results.append(test_basic_coral(X_source, X_target, cat_idx))
    
    # 测试类条件CORAL
    results.append(test_class_conditional_coral(X_source, y_source, X_target, y_target, cat_idx))
    
    # 比较CORAL方法
    results.append(compare_coral_methods(X_source, y_source, X_target, y_target, cat_idx))
    
    # 输出总结
    logging.info("\n=== 测试总结 ===")
    passed_tests = sum(results)
    total_tests = len(results)
    
    logging.info(f"通过测试: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        logging.info("🎉 所有CORAL功能测试通过！")
        return True
    else:
        logging.error("❌ 部分测试失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 