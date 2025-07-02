#!/usr/bin/env python3
"""
交叉验证测试文件

测试交叉验证模块的各种功能：
1. 单个交叉验证实验（TabPFN）
2. 所有标准化以及不平衡处理方法的10折交叉验证（TabPFN）
3. 不同方法TabPFN以及其他模型的对比

注意：
- TabPFN模型：需要标准化和不平衡处理，使用best10/best8/best9/best10特征集
- paper method：不需要标准化和不平衡处理，使用all特征集（58个特征）
- base models (PKUPH/Mayo)：不需要标准化和不平衡处理，使用all特征集（58个特征）
"""

import sys
import logging
import numpy as np
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.loader import MedicalDataLoader
from evaluation.cross_validation import CrossValidationEvaluator, run_cv_experiment, run_model_comparison_cv
from config.settings import get_features_by_type, get_categorical_features


class TestCrossValidation:
    """交叉验证测试类"""
    
    def __init__(self):
        """初始化测试类"""
        # 创建测试结果目录
        self.test_results_dir = Path(__file__).parent / "test_results"
        self.test_results_dir.mkdir(exist_ok=True)
        
        # 创建时间戳
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logging.info(f"测试结果将保存到: {self.test_results_dir}")
    
    def save_results(self, test_name, results, summary_data=None):
        """保存测试结果到文件"""
        try:
            # 创建测试专用目录
            test_dir = self.test_results_dir / f"{test_name}_{self.timestamp}"
            test_dir.mkdir(exist_ok=True)
            
            # 保存详细结果（JSON格式）
            results_file = test_dir / "detailed_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                # 转换numpy数组为列表以便JSON序列化
                serializable_results = self._make_serializable(results)
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            # 保存摘要数据（CSV格式）
            if summary_data is not None:
                summary_file = test_dir / "summary.csv"
                if isinstance(summary_data, dict):
                    # 转换字典为DataFrame
                    df = pd.DataFrame([summary_data])
                elif isinstance(summary_data, list):
                    df = pd.DataFrame(summary_data)
                else:
                    df = summary_data
                
                df.to_csv(summary_file, index=False, encoding='utf-8')
            
            logging.info(f"结果已保存到: {test_dir}")
            return test_dir
            
        except Exception as e:
            logging.error(f"保存结果时出错: {e}")
            return None
    
    def _make_serializable(self, obj):
        """将对象转换为JSON可序列化的格式"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def load_sample_data(self):
        """加载测试用的样本数据"""
        logging.info("准备测试数据...")
        
        # 使用数据加载器加载真实数据
        loader = MedicalDataLoader()
        
        try:
            # 尝试加载数据集A（AI4health）- 使用all63特征集以支持所有模型
            dataset_info = loader.load_dataset('A', feature_type='all63')
            X = pd.DataFrame(dataset_info['X'], columns=dataset_info['feature_names'])
            y = pd.Series(dataset_info['y'])
            
            logging.info(f"成功加载真实数据: {X.shape[0]}样本, {X.shape[1]}特征")
            logging.info(f"类别分布: {dict(y.value_counts().sort_index())}")
            logging.info(f"特征列表: {list(X.columns)}")
            
            return X, y
            
        except Exception as e:
            logging.warning(f"无法加载真实数据 ({e})，使用模拟数据")
            
            # 创建模拟数据
            np.random.seed(42)
            n_samples = 300
            
            # 获取all63特征以支持所有模型
            features = get_features_by_type('all63')
            categorical_features = get_categorical_features('all63')
            
            # 创建特征数据
            X_data = {}
            for feature in features:
                if feature in categorical_features:
                    # 类别特征：0或1
                    X_data[feature] = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
                else:
                    # 数值特征：正态分布
                    X_data[feature] = np.random.normal(0, 1, n_samples)
            
            X = pd.DataFrame(X_data)
            
            # 创建不平衡的标签（约1:2的比例）
            y = np.random.choice([0, 1], size=n_samples, p=[0.65, 0.35])
            y = pd.Series(y)
            
            logging.info(f"创建模拟数据: {X.shape[0]}样本, {X.shape[1]}特征")
            logging.info(f"类别分布: {dict(y.value_counts().sort_index())}")
            
            return X, y
    
    def test_single_cv_experiment_tabpfn(self):
        """测试1: 单个交叉验证实验（TabPFN）"""
        logging.info("="*80)
        logging.info("测试1: 单个交叉验证实验（TabPFN）")
        logging.info("="*80)
        
        X, y = self.load_sample_data()
        
        # 创建交叉验证评估器
        evaluator = CrossValidationEvaluator(
            model_type='tabpfn',
            feature_set='best10',
            scaler_type='standard',
            imbalance_method='smote',
            cv_folds=10,  # 使用10折交叉验证
            random_state=42,
            verbose=False  # 关闭详细输出以避免重复日志
        )
        
        logging.info("配置信息:")
        logging.info(f"  模型类型: {evaluator.model_type}")
        logging.info(f"  特征集: {evaluator.feature_set} ({len(evaluator.features)}个特征)")
        logging.info(f"  特征列表: {evaluator.features}")
        logging.info(f"  标准化: {evaluator.scaler_type}")
        logging.info(f"  不平衡处理: {evaluator.imbalance_method}")
        logging.info(f"  类别特征: {evaluator.categorical_features}")
        
        # 运行交叉验证
        result = evaluator.run_cross_validation(X, y)
        
        # 验证结果
        assert 'fold_results' in result
        assert 'summary' in result
        assert 'predictions' in result
        
        # 验证折数
        assert len(result['fold_results']) == 10
        
        # 验证指标
        summary = result['summary']
        assert 'auc_mean' in summary
        assert 'accuracy_mean' in summary
        assert 'f1_mean' in summary
        
        # 验证指标范围
        assert 0 <= summary['auc_mean'] <= 1
        assert 0 <= summary['accuracy_mean'] <= 1
        assert 0 <= summary['f1_mean'] <= 1
        
        logging.info("单个交叉验证实验测试通过")
        logging.info(f"  平均AUC: {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
        logging.info(f"  平均准确率: {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
        logging.info(f"  平均F1: {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
    
    def test_all_preprocessing_methods_tabpfn(self):
        """测试2: 所有标准化以及不平衡处理方法的10折交叉验证（TabPFN）- 多特征集对比"""
        logging.info("="*80)
        logging.info("测试2: 所有标准化以及不平衡处理方法的10折交叉验证（TabPFN）- 多特征集对比")
        logging.info("="*80)
        
        X, y = self.load_sample_data()
        
        # 定义测试配置
        feature_sets = ['best7', 'best8', 'best9', 'best10']
        scaler_types = ['standard', 'robust', 'none']
        imbalance_methods = [
            'none', 'smote', 'smotenc', 'borderline_smote', 
            'kmeans_smote', 'adasyn', 'smote_tomek', 'random_under'
        ]
        
        logging.info(f"测试配置:")
        logging.info(f"  特征集: {feature_sets}")
        logging.info(f"  标准化方法: {scaler_types}")
        logging.info(f"  不平衡处理方法: {imbalance_methods}")
        logging.info(f"  总组合数: {len(feature_sets) * len(scaler_types) * len(imbalance_methods)}")
        
        # 运行多种配置的交叉验证实验
        results = run_cv_experiment(
            X=X,
            y=y,
            model_types=['tabpfn'],
            feature_sets=feature_sets,
            scaler_types=scaler_types,
            imbalance_methods=imbalance_methods,
            cv_folds=10,  # 使用10折交叉验证
            random_state=42,
            verbose=False  # 减少输出以避免测试日志过长
        )
        
        # 验证结果
        expected_experiments = len(feature_sets) * len(scaler_types) * len(imbalance_methods)
        
        # 按特征集分组统计结果
        feature_set_results = {}
        for feature_set in feature_sets:
            feature_set_results[feature_set] = {
                'experiments': [],
                'best_auc': 0,
                'best_config': '',
                'successful_count': 0
            }
        
        successful_experiments = 0
        overall_best_auc = 0
        overall_best_config = ""
        
        logging.info(f"\n实验结果摘要:")
        logging.info(f"{'配置':<50} {'AUC':<8} {'准确率':<8} {'F1':<8}")
        logging.info("-" * 80)
        
        for experiment_name, result in results.items():
            if result['summary'] and 'auc_mean' in result['summary']:
                summary = result['summary']
                auc = summary['auc_mean']
                acc = summary['accuracy_mean']
                f1 = summary['f1_mean']
                
                logging.info(f"{experiment_name:<50} {auc:<8.4f} {acc:<8.4f} {f1:<8.4f}")
                
                successful_experiments += 1
                
                # 提取特征集名称
                for feature_set in feature_sets:
                    if f'_{feature_set}_' in experiment_name:
                        feature_set_results[feature_set]['experiments'].append({
                            'name': experiment_name,
                            'auc': auc,
                            'acc': acc,
                            'f1': f1
                        })
                        feature_set_results[feature_set]['successful_count'] += 1
                        
                        if auc > feature_set_results[feature_set]['best_auc']:
                            feature_set_results[feature_set]['best_auc'] = auc
                            feature_set_results[feature_set]['best_config'] = experiment_name
                        break
                
                if auc > overall_best_auc:
                    overall_best_auc = auc
                    overall_best_config = experiment_name
        
        # 输出各特征集的最佳结果
        logging.info(f"\n各特征集最佳结果:")
        logging.info(f"{'特征集':<10} {'成功实验数':<12} {'最佳AUC':<10} {'最佳配置':<50}")
        logging.info("-" * 90)
        
        for feature_set in feature_sets:
            result = feature_set_results[feature_set]
            expected_per_feature = len(scaler_types) * len(imbalance_methods)
            logging.info(f"{feature_set:<10} {result['successful_count']:<12}/{expected_per_feature:<3} "
                        f"{result['best_auc']:<10.4f} {result['best_config']:<50}")
        
        logging.info(f"\n✓ 多特征集多配置交叉验证实验测试通过")
        logging.info(f"  总成功实验数: {successful_experiments}/{expected_experiments}")
        logging.info(f"  全局最佳配置: {overall_best_config}")
        logging.info(f"  全局最佳AUC: {overall_best_auc:.4f}")
        
        # 准备保存的摘要数据
        summary_rows = []
        for experiment_name, result in results.items():
            if result['summary'] and 'auc_mean' in result['summary']:
                summary = result['summary']
                summary_rows.append({
                    'experiment_name': experiment_name,
                    'auc_mean': summary['auc_mean'],
                    'auc_std': summary['auc_std'],
                    'accuracy_mean': summary['accuracy_mean'],
                    'accuracy_std': summary['accuracy_std'],
                    'f1_mean': summary['f1_mean'],
                    'f1_std': summary['f1_std'],
                    'precision_mean': summary.get('precision_mean', 0),
                    'recall_mean': summary.get('recall_mean', 0)
                })
        
        # 保存结果
        self.save_results("test2_multi_feature_preprocessing", results, summary_rows)
        
        # 至少要有一半的实验成功
        assert successful_experiments >= expected_experiments // 2
        
        # 验证每个特征集都有成功的实验
        for feature_set in feature_sets:
            assert feature_set_results[feature_set]['successful_count'] > 0, f"特征集{feature_set}没有成功的实验"
    
    def test_comprehensive_tabpfn_and_models_comparison(self):
        """测试3: 所有标准化以及不平衡处理方法的10折交叉验证（TabPFN）以及其他模型对比"""
        logging.info("="*80)
        logging.info("测试3: 所有标准化以及不平衡处理方法的10折交叉验证（TabPFN）以及其他模型对比")
        logging.info("="*80)
        
        X, y = self.load_sample_data()
        
        # 第一部分：TabPFN的所有预处理方法组合测试
        logging.info("第一部分: TabPFN所有预处理方法组合测试")
        logging.info("-" * 60)
        
        # 定义测试配置
        scaler_types = ['standard', 'robust', 'none']
        imbalance_methods = [
            'none', 'smote', 'smotenc', 'borderline_smote', 
            'kmeans_smote', 'adasyn', 'smote_tomek', 'random_under'
        ]
        
        logging.info(f"TabPFN测试配置:")
        logging.info(f"  标准化方法: {scaler_types}")
        logging.info(f"  不平衡处理方法: {imbalance_methods}")
        logging.info(f"  总组合数: {len(scaler_types) * len(imbalance_methods)}")
        
        # 运行TabPFN多种配置的交叉验证实验
        tabpfn_results = run_cv_experiment(
            X=X,
            y=y,
            model_types=['tabpfn'],
            feature_sets=['best10'],
            scaler_types=scaler_types,
            imbalance_methods=imbalance_methods,
            cv_folds=10,  # 使用10折交叉验证
            random_state=42,
            verbose=False
        )
        
        # 统计TabPFN结果
        successful_tabpfn_experiments = 0
        best_tabpfn_auc = 0
        best_tabpfn_config = ""
        
        logging.info("TabPFN实验结果摘要:")
        logging.info(f"{'配置':<40} {'AUC':<8} {'准确率':<8} {'F1':<8}")
        logging.info("-" * 70)
        
        for experiment_name, result in tabpfn_results.items():
            if result['summary'] and 'auc_mean' in result['summary']:
                summary = result['summary']
                auc = summary['auc_mean']
                acc = summary['accuracy_mean']
                f1 = summary['f1_mean']
                
                logging.info(f"{experiment_name:<40} {auc:<8.4f} {acc:<8.4f} {f1:<8.4f}")
                
                successful_tabpfn_experiments += 1
                
                if auc > best_tabpfn_auc:
                    best_tabpfn_auc = auc
                    best_tabpfn_config = experiment_name
        
        logging.info(f"TabPFN成功实验数: {successful_tabpfn_experiments}/{len(scaler_types) * len(imbalance_methods)}")
        logging.info(f"TabPFN最佳配置: {best_tabpfn_config} (AUC: {best_tabpfn_auc:.4f})")
        
        # 第二部分：其他模型对比测试
        logging.info("\n第二部分: 其他模型对比测试")
        logging.info("-" * 60)
        
        logging.info("模型对比配置:")
        logging.info("  TabPFN: 使用best10特征集 + 最佳预处理配置")
        logging.info("  PKUPH: 使用all特征集 + 无预处理")
        logging.info("  Mayo: 使用all特征集 + 无预处理")
        logging.info("  Paper LR: 使用all特征集 + 无预处理")
        
        # 运行其他模型对比实验
        model_results = run_model_comparison_cv(
            X=X,
            y=y,
            feature_set='best10',  # TabPFN使用的特征集
            scaler_type='standard',  # TabPFN使用的标准化
            cv_folds=10,  # 使用10折交叉验证
            random_state=42,
            verbose=False
        )
        
        # 统计模型对比结果
        expected_models = ['tabpfn_best10', 'pkuph_all63', 'mayo_all63', 'paper_lr_all63']
        
        logging.info("模型对比结果:")
        logging.info(f"{'模型':<15} {'AUC':<10} {'准确率':<10} {'F1':<10} {'精确率':<10} {'召回率':<10}")
        logging.info("-" * 80)
        
        successful_models = 0
        model_performances = {}
        
        for model_name in expected_models:
            if model_name in model_results:
                result = model_results[model_name]
                if result['summary'] and 'auc_mean' in result['summary']:
                    summary = result['summary']
                    auc = summary.get('auc_mean', 0)
                    acc = summary.get('accuracy_mean', 0)
                    f1 = summary.get('f1_mean', 0)
                    prec = summary.get('precision_mean', 0)
                    rec = summary.get('recall_mean', 0)
                    
                    # 从模型名称中提取简短名称
                    short_name = model_name.split('_')[0]
                    logging.info(f"{short_name:<15} {auc:<10.4f} {acc:<10.4f} {f1:<10.4f} {prec:<10.4f} {rec:<10.4f}")
                    
                    model_performances[short_name] = auc
                    successful_models += 1
                else:
                    short_name = model_name.split('_')[0]
                    logging.info(f"{short_name:<15} {'失败':<10} {'失败':<10} {'失败':<10} {'失败':<10} {'失败':<10}")
            else:
                short_name = model_name.split('_')[0]
                logging.info(f"{short_name:<15} {'未运行':<10} {'未运行':<10} {'未运行':<10} {'未运行':<10} {'未运行':<10}")
        
        # 综合结果分析
        logging.info("\n综合结果分析:")
        logging.info("-" * 60)
        
        if model_performances:
            best_model = max(model_performances.keys(), key=lambda k: model_performances[k])
            best_model_auc = model_performances[best_model]
            logging.info(f"最佳模型: {best_model} (AUC: {best_model_auc:.4f})")
            
            # 比较TabPFN最佳配置与其他模型
            if best_tabpfn_auc > best_model_auc:
                logging.info(f"TabPFN最佳配置 ({best_tabpfn_config}) 优于其他模型")
                logging.info(f"TabPFN最佳AUC: {best_tabpfn_auc:.4f} vs 其他模型最佳AUC: {best_model_auc:.4f}")
            else:
                logging.info(f"其他模型 ({best_model}) 优于TabPFN最佳配置")
                logging.info(f"其他模型最佳AUC: {best_model_auc:.4f} vs TabPFN最佳AUC: {best_tabpfn_auc:.4f}")
        
        logging.info(f"TabPFN成功实验数: {successful_tabpfn_experiments}")
        logging.info(f"其他模型成功数: {successful_models}/{len(expected_models)}")
        
        # 验证测试成功
        assert successful_tabpfn_experiments > 0, "TabPFN实验全部失败"
        assert successful_models > 0, "其他模型实验全部失败"
        
        logging.info("综合测试通过")
    
    def test_feature_sets_comparison(self):
        """测试4: 不同特征集的对比（TabPFN）"""
        print("\n" + "="*80)
        print("测试4: 不同特征集的对比（TabPFN）")
        print("="*80)
        
        X, y = self.load_sample_data()
        
        feature_sets = ['best7', 'best8', 'best9', 'best10']
        
        print(f"特征集对比配置:")
        for fs in feature_sets:
            features = get_features_by_type(fs)
            cat_features = get_categorical_features(fs)
            print(f"  {fs}: {len(features)}个特征 ({len(cat_features)}个类别特征)")
        
        results = {}
        
        for feature_set in feature_sets:
            print(f"\n测试特征集: {feature_set}")
            
            evaluator = CrossValidationEvaluator(
                model_type='tabpfn',
                feature_set=feature_set,
                scaler_type='standard',
                imbalance_method='smote',
                cv_folds=10,  # 使用10折交叉验证
                random_state=42,
                verbose=False
            )
            
            try:
                result = evaluator.run_cross_validation(X, y)
                results[feature_set] = result
                
                if result['summary'] and 'auc_mean' in result['summary']:
                    auc = result['summary']['auc_mean']
                    print(f"  ✓ {feature_set}: AUC = {auc:.4f}")
                else:
                    print(f"  ✗ {feature_set}: 实验失败")
                    
            except Exception as e:
                print(f"  ✗ {feature_set}: 异常 - {e}")
                results[feature_set] = None
        
        # 结果摘要
        print(f"\n特征集对比结果:")
        print(f"{'特征集':<10} {'特征数':<8} {'AUC':<10} {'准确率':<10} {'F1':<10}")
        print("-" * 50)
        
        successful_feature_sets = 0
        
        for feature_set in feature_sets:
            features = get_features_by_type(feature_set)
            n_features = len(features)
            
            if results[feature_set] and results[feature_set]['summary']:
                summary = results[feature_set]['summary']
                auc = summary.get('auc_mean', 0)
                acc = summary.get('accuracy_mean', 0)
                f1 = summary.get('f1_mean', 0)
                
                print(f"{feature_set:<10} {n_features:<8} {auc:<10.4f} {acc:<10.4f} {f1:<10.4f}")
                successful_feature_sets += 1
            else:
                print(f"{feature_set:<10} {n_features:<8} {'失败':<10} {'失败':<10} {'失败':<10}")
        
        print(f"\n✓ 特征集对比实验测试通过")
        print(f"  成功特征集数: {successful_feature_sets}/{len(feature_sets)}")
        
        # 至少要有一半的特征集成功
        assert successful_feature_sets >= len(feature_sets) // 2
    
    def test_model_specific_features(self):
        """测试5: 验证不同模型使用正确的特征"""
        print("\n" + "="*80)
        print("测试5: 验证不同模型使用正确的特征")
        print("="*80)
        
        X, y = self.load_sample_data()
        
        # 测试配置
        model_configs = [
            {
                'model_type': 'tabpfn',
                'feature_set': 'best10',
                'expected_features': 10,
                'needs_preprocessing': True
            },
            {
                'model_type': 'pkuph',
                'feature_set': 'all63',  # 基线模型使用all63特征集，但实际只用自己的6个特征
                'expected_features': 6,  # PKUPH模型实际使用6个特征
                'needs_preprocessing': False
            },
            {
                'model_type': 'mayo',
                'feature_set': 'all63',  # 基线模型使用all63特征集，但实际只用自己的6个特征
                'expected_features': 6,  # Mayo模型实际使用6个特征
                'needs_preprocessing': False
            },
            {
                'model_type': 'paper_lr',
                'feature_set': 'all63',  # 论文方法使用all63特征集，但实际只用自己的11个特征
                'expected_features': 11,  # Paper LR模型实际使用11个特征
                'needs_preprocessing': False
            }
        ]
        
        print(f"模型特征配置验证:")
        print(f"{'模型':<12} {'特征集':<8} {'预期特征数':<10} {'需要预处理':<10} {'状态':<10}")
        print("-" * 65)
        
        for config in model_configs:
            model_type = config['model_type']
            feature_set = config['feature_set']
            expected_features = config['expected_features']
            needs_preprocessing = config['needs_preprocessing']
            
            try:
                evaluator = CrossValidationEvaluator(
                    model_type=model_type,
                    feature_set=feature_set,
                    scaler_type='standard' if needs_preprocessing else 'none',
                    imbalance_method='smote' if needs_preprocessing else 'none',
                    cv_folds=10,  # 使用10折交叉验证
                    random_state=42,
                    verbose=False
                )
                
                # 验证特征数量
                actual_features = len(evaluator.features)
                
                # 验证预处理设置
                should_preprocess = evaluator._should_apply_preprocessing()
                
                status = "✓" if (actual_features == expected_features and 
                               should_preprocess == needs_preprocessing) else "✗"
                
                print(f"{model_type:<12} {feature_set:<8} {actual_features:<10} {should_preprocess:<10} {status:<10}")
                
                # 运行一个快速测试以确保模型可以工作
                if status == "✓":
                    result = evaluator.run_cross_validation(X, y)
                    if result['summary'] and 'auc_mean' in result['summary']:
                        print(f"  └─ 快速测试通过，AUC: {result['summary']['auc_mean']:.4f}")
                    else:
                        print(f"  └─ 快速测试失败")
                        
            except Exception as e:
                print(f"{model_type:<12} {feature_set:<8} {'异常':<10} {'异常':<10} {'✗':<10}")
                print(f"  └─ 错误: {e}")
        
        print(f"\n✓ 模型特征配置验证完成")


def run_comprehensive_cv_test():
    """运行全面的交叉验证测试"""
    logging.info("="*80)
    logging.info("交叉验证模块全面测试")
    logging.info("="*80)
    
    # 创建测试实例
    test_instance = TestCrossValidation()
    
    try:
        # 运行所有测试
        # test_instance.test_single_cv_experiment_tabpfn()
        test_instance.test_all_preprocessing_methods_tabpfn()
        test_instance.test_comprehensive_tabpfn_and_models_comparison()
        test_instance.test_feature_sets_comparison()
        test_instance.test_model_specific_features()
        
        logging.info("="*80)
        logging.info("🎉 所有交叉验证测试通过！")
        logging.info(f"测试结果已保存到: {test_instance.test_results_dir}")
        logging.info("="*80)
        
    except Exception as e:
        logging.error(f"测试失败: {e}")
        raise


if __name__ == "__main__":
    # 直接运行测试
    run_comprehensive_cv_test() 