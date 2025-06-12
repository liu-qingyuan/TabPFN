#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多模型功能演示脚本

展示如何使用新的多模型支持和跨域实验功能
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def demo_model_selector():
    """演示模型选择器功能"""
    logger = logging.getLogger(__name__)
    logger.info("=== 模型选择器演示 ===")
    
    try:
        from analytical_mmd_A2B_feature58.modeling.model_selector import (
            get_available_models, get_model, validate_model_params
        )
        from analytical_mmd_A2B_feature58.config.settings import (
            get_features_by_type, get_categorical_indices, get_model_config
        )
        
        # 检查可用模型
        available_models = get_available_models()
        logger.info(f"可用模型类型: {available_models}")
        
        # 演示特征配置
        best7_features = get_features_by_type('best7')
        cat_indices = get_categorical_indices('best7')
        logger.info(f"最佳7特征: {best7_features}")
        logger.info(f"类别特征索引: {cat_indices}")
        
        # 创建模拟数据
        np.random.seed(42)
        n_samples, n_features = 100, len(best7_features)
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        # 测试每种可用的模型类型
        for model_type in available_models:
            logger.info(f"\n--- 测试 {model_type.upper()} 模型 ---")
            
            try:
                # 获取模型配置
                config = get_model_config(model_type)  # 移除预设参数
                logger.info(f"模型配置: {config}")
                
                # 验证参数
                validated_config = validate_model_params(model_type, config)
                logger.info(f"验证后配置: {validated_config}")
                
                # 创建模型
                model = get_model(
                    model_type, 
                    categorical_feature_indices=cat_indices,
                    **validated_config
                )
                
                logger.info(f"成功创建 {model_type} 模型: {type(model).__name__}")
                
                # 如果是RF模型，可以进行完整测试
                if model_type == 'rf':
                    logger.info("进行RF模型训练和预测测试...")
                    model.fit(X, y)
                    predictions = model.predict(X)
                    probabilities = model.predict_proba(X)
                    
                    logger.info(f"预测结果形状: {predictions.shape}")
                    logger.info(f"概率结果形状: {probabilities.shape}")
                    logger.info(f"预测准确率: {np.mean(predictions == y):.4f}")
                
            except Exception as e:
                logger.warning(f"{model_type} 模型测试失败: {e}")
                
    except ImportError as e:
        logger.error(f"导入模型选择器失败: {e}")

def demo_config_management():
    """演示配置管理功能"""
    logger = logging.getLogger(__name__)
    logger.info("\n=== 配置管理演示 ===")
    
    try:
        from analytical_mmd_A2B_feature58.config.settings import (
            get_features_by_type, get_categorical_indices, get_model_config,
            MODEL_PRESETS, BEST_7_FEATURES, BEST_7_CAT_IDX
        )
        
        # 特征配置演示
        logger.info("特征配置:")
        for feature_type in ['all', 'best7']:
            features = get_features_by_type(feature_type)
            cat_indices = get_categorical_indices(feature_type)
            logger.info(f"  {feature_type}: {len(features)}个特征, {len(cat_indices)}个类别特征")
        
        # 模型配置演示
        logger.info("\n模型配置:")
        for model_type in ['auto', 'base', 'rf']:
            for preset in ['fast', 'balanced', 'accurate']:
                try:
                    config = get_model_config(model_type, preset)
                    logger.info(f"  {model_type}-{preset}: {len(config)}个参数")
                except:
                    logger.info(f"  {model_type}-{preset}: 配置不可用")
        
        # 预设配置演示
        logger.info(f"\n可用预设: {list(MODEL_PRESETS.keys())}")
        
        # 最佳7特征详情
        logger.info(f"\n最佳7特征详情:")
        logger.info(f"  特征列表: {BEST_7_FEATURES}")
        logger.info(f"  类别特征索引: {BEST_7_CAT_IDX}")
        
    except ImportError as e:
        logger.error(f"导入配置模块失败: {e}")

def demo_cross_domain_runner():
    """演示跨域实验运行器功能"""
    logger = logging.getLogger(__name__)
    logger.info("\n=== 跨域实验运行器演示 ===")
    
    try:
        from analytical_mmd_A2B_feature58.modeling.cross_domain_runner import (
            CrossDomainExperimentRunner
        )
        
        # 创建临时目录
        import tempfile
        temp_dir = tempfile.mkdtemp()
        logger.info(f"使用临时目录: {temp_dir}")
        
        # 创建模拟数据文件
        create_mock_datasets(temp_dir, logger)
        
        # 创建跨域实验运行器
        runner = CrossDomainExperimentRunner(
            model_type='rf',  # 使用RF模型（总是可用）
            feature_type='best7',
            use_mmd_adaptation=True,
            mmd_method='mean_std',  # 使用简单的方法
            use_class_conditional=False,
            use_threshold_optimizer=False,
            save_path=os.path.join(temp_dir, 'results')
        )
        
        logger.info("跨域实验运行器创建成功")
        logger.info(f"  模型类型: {runner.model_type}")
        logger.info(f"  特征类型: {runner.feature_type}")
        logger.info(f"  MMD方法: {runner.mmd_method}")
        logger.info(f"  特征数量: {len(runner.features)}")
        
        # 注意：实际运行需要真实数据文件，这里只演示初始化
        logger.info("注意：完整实验需要真实数据文件")
        
        # 清理临时目录
        import shutil
        shutil.rmtree(temp_dir)
        
    except ImportError as e:
        logger.error(f"导入跨域实验运行器失败: {e}")

def create_mock_datasets(temp_dir, logger):
    """创建模拟数据集文件"""
    try:
        from analytical_mmd_A2B_feature58.config.settings import BEST_7_FEATURES
        
        np.random.seed(42)
        
        # 创建模拟数据集A
        n_samples_a = 200
        X_A = np.random.randn(n_samples_a, len(BEST_7_FEATURES))
        y_A = np.random.randint(0, 2, n_samples_a)
        
        df_A = pd.DataFrame(X_A, columns=BEST_7_FEATURES)
        df_A['Label'] = y_A
        df_A.to_excel(os.path.join(temp_dir, 'dataset_A.xlsx'), index=False)
        
        # 创建模拟数据集B
        n_samples_b = 150
        X_B = np.random.randn(n_samples_b, len(BEST_7_FEATURES))
        y_B = np.random.randint(0, 2, n_samples_b)
        
        df_B = pd.DataFrame(X_B, columns=BEST_7_FEATURES)
        df_B['Label'] = y_B
        df_B.to_excel(os.path.join(temp_dir, 'dataset_B.xlsx'), index=False)
        
        logger.info("模拟数据集创建成功")
        
    except Exception as e:
        logger.warning(f"创建模拟数据集失败: {e}")

def demo_command_line_usage():
    """演示命令行使用方法"""
    logger = logging.getLogger(__name__)
    logger.info("\n=== 命令行使用演示 ===")
    
    examples = [
        {
            "描述": "基本跨域实验",
            "命令": "python scripts/run_analytical_mmd.py --mode cross-domain --model-type rf --method mean_std"
        },
        {
            "描述": "使用最佳7特征 AutoTabPFN",
            "命令": "python scripts/run_analytical_mmd.py --mode cross-domain --model-type auto --feature-type best7 --method linear"
        },
        {
            "描述": "使用TunedTabPFN（超参数优化）",
            "命令": "python scripts/run_analytical_mmd.py --mode cross-domain --model-type tuned --feature-type best7 --method linear"
        },
        {
            "描述": "完整功能组合",
            "命令": "python scripts/run_analytical_mmd.py --mode cross-domain --model-type auto --feature-type best7 --method linear --use-class-conditional --use-threshold-optimizer"
        },
        {
            "描述": "快速测试模式",
            "命令": "python scripts/run_analytical_mmd.py --mode cross-domain --model-type rf --method mean_std --no-visualizations"
        },
        {
            "描述": "比较所有方法",
            "命令": "python scripts/run_analytical_mmd.py --mode cross-domain --model-type auto --compare-all"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        logger.info(f"\n{i}. {example['描述']}:")
        logger.info(f"   {example['命令']}")

def main():
    """主函数"""
    logger = setup_logging()
    
    logger.info("多模型功能演示开始")
    logger.info("=" * 50)
    
    try:
        # 演示各个功能模块
        demo_model_selector()
        demo_config_management()
        demo_cross_domain_runner()
        demo_command_line_usage()
        
        logger.info("\n" + "=" * 50)
        logger.info("演示完成！")
        
        logger.info("\n下一步:")
        logger.info("1. 安装所需依赖: pip install tabpfn tabpfn_extensions")
        logger.info("2. 准备数据文件到 data/ 目录")
        logger.info("3. 运行完整实验: python scripts/run_analytical_mmd.py --mode cross-domain --model-type auto --feature-type best7 --method linear")
        
    except Exception as e:
        logger.error(f"演示过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 