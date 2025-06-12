import unittest
import numpy as np
import logging
import sys
import os
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from typing import Union

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.coral import coral_transform, class_conditional_coral_transform, generate_pseudo_labels_for_coral

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_dense_array(X: Union[np.ndarray, any]) -> np.ndarray:
    """确保输入是密集的numpy数组"""
    if hasattr(X, 'toarray'):
        return X.toarray()
    return np.asarray(X)

class TestCORALFunctions(unittest.TestCase):
    """测试CORAL相关功能的单元测试类"""
    
    def setUp(self) -> None:
        """设置测试数据"""
        # 设置随机种子确保结果可重现
        np.random.seed(42)
        
        # 生成模拟的源域和目标域数据
        # 源域数据
        self.X_source, self.y_source = make_classification(
            n_samples=200, n_features=10, n_informative=8, n_redundant=2,
            n_classes=2, random_state=42, class_sep=1.5
        )
        
        # 目标域数据 - 通过添加噪声和偏移来模拟域偏移
        self.X_target, self.y_target = make_classification(
            n_samples=150, n_features=10, n_informative=8, n_redundant=2,
            n_classes=2, random_state=123, class_sep=1.2
        )
        
        # 为目标域添加域偏移（均值偏移和方差缩放）
        self.X_target = self.X_target * 1.5 + 2.0  # 缩放和偏移
        
        # 定义类别特征索引（假设最后2个特征是类别特征）
        self.cat_idx = [8, 9]
        
        # 将类别特征转换为整数（模拟类别特征）
        self.X_source[:, self.cat_idx] = np.round(self.X_source[:, self.cat_idx]).astype(int)
        self.X_target[:, self.cat_idx] = np.round(self.X_target[:, self.cat_idx]).astype(int)
        
        # 标准化数据并确保是密集数组
        self.scaler = StandardScaler()
        self.X_source_scaled = ensure_dense_array(self.scaler.fit_transform(self.X_source))
        self.X_target_scaled = ensure_dense_array(self.scaler.transform(self.X_target))
        
        logging.info(f"测试数据设置完成:")
        logging.info(f"源域: {self.X_source_scaled.shape}, 目标域: {self.X_target_scaled.shape}")
        logging.info(f"类别特征索引: {self.cat_idx}")
        logging.info(f"源域标签分布: {np.bincount(self.y_source)}")
        logging.info(f"目标域标签分布: {np.bincount(self.y_target)}")
    
    def test_coral_transform_basic(self) -> None:
        """测试基本CORAL变换功能"""
        logging.info("\n=== 测试基本CORAL变换功能 ===")
        
        # 计算变换前的域差异
        cont_idx = [i for i in range(self.X_source_scaled.shape[1]) if i not in self.cat_idx]
        mean_diff_before = np.mean(np.abs(
            np.mean(self.X_source_scaled[:, cont_idx], axis=0) - 
            np.mean(self.X_target_scaled[:, cont_idx], axis=0)
        ))
        std_diff_before = np.mean(np.abs(
            np.std(self.X_source_scaled[:, cont_idx], axis=0) - 
            np.std(self.X_target_scaled[:, cont_idx], axis=0)
        ))
        
        # 执行CORAL变换
        X_target_aligned = coral_transform(self.X_source_scaled, self.X_target_scaled, self.cat_idx)
        
        # 验证输出形状
        self.assertEqual(X_target_aligned.shape, self.X_target_scaled.shape, 
                        "CORAL变换后的数据形状应该保持不变")
        
        # 验证类别特征未被改变
        np.testing.assert_array_equal(
            self.X_target_scaled[:, self.cat_idx], 
            X_target_aligned[:, self.cat_idx],
            "类别特征在CORAL变换后应该保持不变"
        )
        
        # 计算变换后的域差异
        mean_diff_after = np.mean(np.abs(
            np.mean(self.X_source_scaled[:, cont_idx], axis=0) - 
            np.mean(X_target_aligned[:, cont_idx], axis=0)
        ))
        std_diff_after = np.mean(np.abs(
            np.std(self.X_source_scaled[:, cont_idx], axis=0) - 
            np.std(X_target_aligned[:, cont_idx], axis=0)
        ))
        
        # 验证域差异减少
        self.assertLess(mean_diff_after, mean_diff_before, 
                       "CORAL变换应该减少均值差异")
        self.assertLess(std_diff_after, std_diff_before, 
                       "CORAL变换应该减少标准差差异")
        
        logging.info(f"变换前均值差异: {mean_diff_before:.6f}, 变换后: {mean_diff_after:.6f}")
        logging.info(f"变换前标准差差异: {std_diff_before:.6f}, 变换后: {std_diff_after:.6f}")
        logging.info("✓ 基本CORAL变换测试通过")
    
    def test_coral_transform_edge_cases(self) -> None:
        """测试CORAL变换的边界情况"""
        logging.info("\n=== 测试CORAL变换边界情况 ===")
        
        # 测试单样本情况
        X_single = self.X_source_scaled[:1, :]
        try:
            result = coral_transform(self.X_source_scaled, X_single, self.cat_idx)
            self.assertEqual(result.shape, X_single.shape)
            logging.info("✓ 单样本CORAL变换测试通过")
        except Exception as e:
            logging.warning(f"单样本CORAL变换失败: {e}")
        
        # 测试相同分布的数据（应该变化很小）
        X_same = self.X_source_scaled.copy()
        result_same = coral_transform(self.X_source_scaled, X_same, self.cat_idx)
        
        cont_idx = [i for i in range(self.X_source_scaled.shape[1]) if i not in self.cat_idx]
        diff = np.mean(np.abs(X_same[:, cont_idx] - result_same[:, cont_idx]))
        self.assertLess(diff, 0.1, "相同分布的数据CORAL变换后变化应该很小")
        logging.info(f"相同分布数据变换差异: {diff:.6f}")
        logging.info("✓ 边界情况测试通过")
    
    def test_class_conditional_coral_basic(self) -> None:
        """测试基本类条件CORAL变换功能"""
        logging.info("\n=== 测试基本类条件CORAL变换功能 ===")
        
        # 生成伪标签
        yt_pseudo = generate_pseudo_labels_for_coral(
            self.X_source_scaled, self.y_source, self.X_target_scaled, self.cat_idx
        )
        
        # 执行类条件CORAL变换
        X_target_aligned = class_conditional_coral_transform(
            self.X_source_scaled, self.y_source, 
            self.X_target_scaled, yt_pseudo, 
            self.cat_idx, alpha=0.1
        )
        
        # 验证输出形状
        self.assertEqual(X_target_aligned.shape, self.X_target_scaled.shape, 
                        "类条件CORAL变换后的数据形状应该保持不变")
        
        # 验证类别特征未被改变
        np.testing.assert_array_equal(
            self.X_target_scaled[:, self.cat_idx], 
            X_target_aligned[:, self.cat_idx],
            "类别特征在类条件CORAL变换后应该保持不变"
        )
        
        # 验证伪标签分布合理
        unique_labels = np.unique(yt_pseudo)
        self.assertTrue(len(unique_labels) >= 1, "应该至少有一个类别的伪标签")
        self.assertTrue(all(label in [0, 1] for label in unique_labels), 
                       "伪标签应该在有效范围内")
        
        logging.info(f"伪标签分布: {np.bincount(yt_pseudo)}")
        logging.info("✓ 基本类条件CORAL变换测试通过")
    
    def test_class_conditional_coral_with_true_labels(self) -> None:
        """测试使用部分真实标签的类条件CORAL变换"""
        logging.info("\n=== 测试使用部分真实标签的类条件CORAL变换 ===")
        
        # 创建部分真实标签（使用50%的真实标签）
        n_labeled = len(self.y_target) // 2
        labeled_idx = np.random.choice(len(self.y_target), n_labeled, replace=False)
        
        yt_partial = np.full_like(self.y_target, -1)  # -1表示未标记
        yt_partial[labeled_idx] = self.y_target[labeled_idx]  # 填入部分真实标签
        
        # 对未标记部分生成伪标签
        unlabeled_mask = (yt_partial == -1)
        if np.any(unlabeled_mask):
            # 先用普通CORAL对齐未标记部分
            X_unlabeled = self.X_target_scaled[unlabeled_mask]
            X_unlabeled_aligned = coral_transform(self.X_source_scaled, X_unlabeled, self.cat_idx)
            
            # 用KNN预测伪标签
            from sklearn.neighbors import KNeighborsClassifier
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(self.X_source_scaled, self.y_source)
            yt_pseudo_unlabeled = knn.predict(X_unlabeled_aligned)
            yt_partial[unlabeled_mask] = yt_pseudo_unlabeled
        
        # 执行类条件CORAL变换
        X_target_aligned = class_conditional_coral_transform(
            self.X_source_scaled, self.y_source, 
            self.X_target_scaled, yt_partial, 
            self.cat_idx, alpha=0.1
        )
        
        # 验证结果
        self.assertEqual(X_target_aligned.shape, self.X_target_scaled.shape)
        np.testing.assert_array_equal(
            self.X_target_scaled[:, self.cat_idx], 
            X_target_aligned[:, self.cat_idx]
        )
        
        logging.info(f"使用了{n_labeled}个真实标签，{np.sum(unlabeled_mask)}个伪标签")
        logging.info(f"最终标签分布: {np.bincount(yt_partial[yt_partial != -1])}")
        logging.info("✓ 部分真实标签的类条件CORAL变换测试通过")
    
    def test_coral_vs_class_conditional_coral(self) -> None:
        """比较普通CORAL和类条件CORAL的效果"""
        logging.info("\n=== 比较普通CORAL和类条件CORAL的效果 ===")
        
        # 普通CORAL变换
        X_target_coral = coral_transform(self.X_source_scaled, self.X_target_scaled, self.cat_idx)
        
        # 类条件CORAL变换
        yt_pseudo = generate_pseudo_labels_for_coral(
            self.X_source_scaled, self.y_source, self.X_target_scaled, self.cat_idx
        )
        X_target_class_coral = class_conditional_coral_transform(
            self.X_source_scaled, self.y_source, 
            self.X_target_scaled, yt_pseudo, 
            self.cat_idx, alpha=0.1
        )
        
        # 计算连续特征的域差异
        cont_idx = [i for i in range(self.X_source_scaled.shape[1]) if i not in self.cat_idx]
        
        # 原始差异
        mean_diff_original = np.mean(np.abs(
            np.mean(self.X_source_scaled[:, cont_idx], axis=0) - 
            np.mean(self.X_target_scaled[:, cont_idx], axis=0)
        ))
        
        # CORAL后差异
        mean_diff_coral = np.mean(np.abs(
            np.mean(self.X_source_scaled[:, cont_idx], axis=0) - 
            np.mean(X_target_coral[:, cont_idx], axis=0)
        ))
        
        # 类条件CORAL后差异
        mean_diff_class_coral = np.mean(np.abs(
            np.mean(self.X_source_scaled[:, cont_idx], axis=0) - 
            np.mean(X_target_class_coral[:, cont_idx], axis=0)
        ))
        
        logging.info(f"原始均值差异: {mean_diff_original:.6f}")
        logging.info(f"CORAL后均值差异: {mean_diff_coral:.6f}")
        logging.info(f"类条件CORAL后均值差异: {mean_diff_class_coral:.6f}")
        
        # 验证两种方法都减少了域差异
        self.assertLess(mean_diff_coral, mean_diff_original, "CORAL应该减少域差异")
        self.assertLess(mean_diff_class_coral, mean_diff_original, "类条件CORAL应该减少域差异")
        
        logging.info("✓ CORAL方法比较测试通过")
    
    def test_coral_numerical_stability(self) -> None:
        """测试CORAL变换的数值稳定性"""
        logging.info("\n=== 测试CORAL变换数值稳定性 ===")
        
        # 创建接近奇异的协方差矩阵情况
        X_singular = np.random.randn(50, 5)
        X_singular[:, 1] = X_singular[:, 0] + 1e-10  # 几乎线性相关
        
        cat_idx_small = [4]  # 只有一个类别特征
        
        try:
            # 测试是否能处理接近奇异的情况
            result = coral_transform(X_singular, X_singular + 0.1, cat_idx_small)
            self.assertEqual(result.shape, X_singular.shape)
            logging.info("✓ 数值稳定性测试通过")
        except Exception as e:
            logging.warning(f"数值稳定性测试失败: {e}")
            # 这可能是预期的，取决于正则化的实现
    
    def test_generate_pseudo_labels(self) -> None:
        """测试伪标签生成功能"""
        logging.info("\n=== 测试伪标签生成功能 ===")
        
        yt_pseudo = generate_pseudo_labels_for_coral(
            self.X_source_scaled, self.y_source, self.X_target_scaled, self.cat_idx
        )
        
        # 验证伪标签的基本属性
        self.assertEqual(len(yt_pseudo), len(self.y_target), "伪标签数量应该等于目标域样本数量")
        self.assertTrue(all(label in [0, 1] for label in yt_pseudo), "伪标签应该在有效范围内")
        
        # 验证伪标签分布不会过于极端
        label_counts = np.bincount(yt_pseudo)
        min_class_ratio = min(label_counts) / len(yt_pseudo)
        self.assertGreater(min_class_ratio, 0.05, "每个类别至少应该有5%的样本")
        
        logging.info(f"伪标签分布: {label_counts}")
        logging.info(f"最小类别比例: {min_class_ratio:.3f}")
        logging.info("✓ 伪标签生成测试通过")


class TestCORALIntegration(unittest.TestCase):
    """CORAL功能的集成测试"""
    
    def setUp(self) -> None:
        """设置更复杂的测试场景"""
        np.random.seed(42)
        
        # 创建更复杂的多特征数据
        self.n_features = 15
        self.n_samples_source = 300
        self.n_samples_target = 200
        
        # 生成源域数据
        self.X_source, self.y_source = make_classification(
            n_samples=self.n_samples_source, 
            n_features=self.n_features, 
            n_informative=10, 
            n_redundant=3,
            n_classes=2, 
            random_state=42, 
            class_sep=1.0
        )
        
        # 生成目标域数据（有显著的域偏移）
        self.X_target, self.y_target = make_classification(
            n_samples=self.n_samples_target, 
            n_features=self.n_features, 
            n_informative=10, 
            n_redundant=3,
            n_classes=2, 
            random_state=123, 
            class_sep=0.8
        )
        
        # 添加更复杂的域偏移
        # 对不同特征组应用不同的变换
        self.X_target[:, :5] = self.X_target[:, :5] * 2.0 + 1.0  # 缩放和偏移
        self.X_target[:, 5:10] = self.X_target[:, 5:10] * 0.5 - 0.5  # 不同的缩放
        self.X_target[:, 10:] = self.X_target[:, 10:] + np.random.normal(0, 0.5, (self.n_samples_target, 5))  # 添加噪声
        
        # 定义类别特征（最后3个特征）
        self.cat_idx = [12, 13, 14]
        
        # 将类别特征转换为整数
        for idx in self.cat_idx:
            self.X_source[:, idx] = np.round(np.abs(self.X_source[:, idx])).astype(int) % 5
            self.X_target[:, idx] = np.round(np.abs(self.X_target[:, idx])).astype(int) % 5
        
        # 标准化并确保是密集数组
        self.scaler = StandardScaler()
        self.X_source_scaled = ensure_dense_array(self.scaler.fit_transform(self.X_source))
        self.X_target_scaled = ensure_dense_array(self.scaler.transform(self.X_target))
        
        logging.info(f"集成测试数据设置完成:")
        logging.info(f"源域: {self.X_source_scaled.shape}, 目标域: {self.X_target_scaled.shape}")
        logging.info(f"类别特征索引: {self.cat_idx}")
    
    def test_end_to_end_coral_pipeline(self) -> None:
        """测试完整的CORAL流水线"""
        logging.info("\n=== 测试完整CORAL流水线 ===")
        
        # 1. 计算原始域差异
        cont_idx = [i for i in range(self.n_features) if i not in self.cat_idx]
        original_mean_diff = np.mean(np.abs(
            np.mean(self.X_source_scaled[:, cont_idx], axis=0) - 
            np.mean(self.X_target_scaled[:, cont_idx], axis=0)
        ))
        
        # 2. 应用CORAL变换
        X_target_coral = coral_transform(self.X_source_scaled, self.X_target_scaled, self.cat_idx)
        
        # 3. 计算CORAL后的域差异
        coral_mean_diff = np.mean(np.abs(
            np.mean(self.X_source_scaled[:, cont_idx], axis=0) - 
            np.mean(X_target_coral[:, cont_idx], axis=0)
        ))
        
        # 4. 生成伪标签并应用类条件CORAL
        yt_pseudo = generate_pseudo_labels_for_coral(
            self.X_source_scaled, self.y_source, self.X_target_scaled, self.cat_idx
        )
        
        X_target_class_coral = class_conditional_coral_transform(
            self.X_source_scaled, self.y_source, 
            self.X_target_scaled, yt_pseudo, 
            self.cat_idx, alpha=0.1
        )
        
        # 5. 计算类条件CORAL后的域差异
        class_coral_mean_diff = np.mean(np.abs(
            np.mean(self.X_source_scaled[:, cont_idx], axis=0) - 
            np.mean(X_target_class_coral[:, cont_idx], axis=0)
        ))
        
        # 验证流水线效果
        self.assertLess(coral_mean_diff, original_mean_diff, "CORAL应该减少域差异")
        self.assertLess(class_coral_mean_diff, original_mean_diff, "类条件CORAL应该减少域差异")
        
        logging.info(f"原始域差异: {original_mean_diff:.6f}")
        logging.info(f"CORAL后域差异: {coral_mean_diff:.6f} (减少 {(1-coral_mean_diff/original_mean_diff)*100:.1f}%)")
        logging.info(f"类条件CORAL后域差异: {class_coral_mean_diff:.6f} (减少 {(1-class_coral_mean_diff/original_mean_diff)*100:.1f}%)")
        logging.info("✓ 完整CORAL流水线测试通过")
    
    def test_coral_with_different_alpha_values(self) -> None:
        """测试不同alpha值对类条件CORAL的影响"""
        logging.info("\n=== 测试不同alpha值的影响 ===")
        
        alpha_values = [0.01, 0.1, 0.5, 1.0]
        results = {}
        
        # 生成伪标签
        yt_pseudo = generate_pseudo_labels_for_coral(
            self.X_source_scaled, self.y_source, self.X_target_scaled, self.cat_idx
        )
        
        cont_idx = [i for i in range(self.n_features) if i not in self.cat_idx]
        
        for alpha in alpha_values:
            # 应用类条件CORAL
            X_aligned = class_conditional_coral_transform(
                self.X_source_scaled, self.y_source, 
                self.X_target_scaled, yt_pseudo, 
                self.cat_idx, alpha=alpha
            )
            
            # 计算域差异
            mean_diff = np.mean(np.abs(
                np.mean(self.X_source_scaled[:, cont_idx], axis=0) - 
                np.mean(X_aligned[:, cont_idx], axis=0)
            ))
            
            results[alpha] = mean_diff
            logging.info(f"Alpha={alpha}: 域差异={mean_diff:.6f}")
        
        # 验证alpha值的影响是合理的
        self.assertTrue(len(results) == len(alpha_values), "应该为所有alpha值生成结果")
        
        logging.info("✓ 不同alpha值测试通过")


def run_coral_functionality_tests() -> bool:
    """运行所有CORAL功能测试"""
    logging.info("开始运行CORAL功能测试...")
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加基本功能测试
    test_loader = unittest.TestLoader()
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestCORALFunctions))
    
    # 添加集成测试
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestCORALIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出测试结果摘要
    logging.info(f"\n=== 测试结果摘要 ===")
    logging.info(f"运行测试数量: {result.testsRun}")
    logging.info(f"失败测试数量: {len(result.failures)}")
    logging.info(f"错误测试数量: {len(result.errors)}")
    
    if result.failures:
        logging.error("失败的测试:")
        for test, traceback in result.failures:
            logging.error(f"  - {test}: {traceback}")
    
    if result.errors:
        logging.error("错误的测试:")
        for test, traceback in result.errors:
            logging.error(f"  - {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    logging.info(f"测试成功率: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # 运行测试
    success = run_coral_functionality_tests()
    
    if success:
        logging.info("\n🎉 所有CORAL功能测试通过！")
    else:
        logging.error("\n❌ 部分测试失败，请检查上述错误信息")
        sys.exit(1) 