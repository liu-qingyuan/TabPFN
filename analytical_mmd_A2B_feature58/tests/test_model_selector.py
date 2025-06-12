#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的模型选择器测试脚本

用于验证新的模型选择器是否正常工作
"""

import sys
import os
import logging
import numpy as np

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

def test_model_selector():
    """测试模型选择器功能"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("测试模型选择器功能...")
    
    try:
        from modeling.model_selector import get_available_models, get_model, test_model_availability
        from config.settings import BEST_7_FEATURES, BEST_7_CAT_IDX
        
        # 测试模型可用性
        logger.info("检查模型可用性...")
        available_models = get_available_models()
        logger.info(f"可用模型: {available_models}")
        
        # 详细测试
        results = test_model_availability()
        logger.info(f"详细测试结果: {results}")
        
        # 创建测试数据
        np.random.seed(42)
        n_samples, n_features = 50, len(BEST_7_FEATURES)
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        # 测试每种可用模型
        for model_type in available_models:
            logger.info(f"\n--- 测试 {model_type} 模型 ---")
            
            try:
                # 创建模型
                model = get_model(model_type, categorical_feature_indices=BEST_7_CAT_IDX)
                logger.info(f"成功创建模型: {type(model).__name__}")
                
                # 如果是RF模型，进行完整测试
                if model_type == 'rf':
                    logger.info("进行训练和预测测试...")
                    model.fit(X, y)
                    predictions = model.predict(X)
                    probabilities = model.predict_proba(X)
                    
                    logger.info(f"预测形状: {predictions.shape}")
                    logger.info(f"概率形状: {probabilities.shape}")
                    logger.info(f"准确率: {np.mean(predictions == y):.4f}")
                else:
                    logger.info("模型创建成功（跳过训练测试）")
                    
            except Exception as e:
                logger.error(f"{model_type} 模型测试失败: {e}")
        
        logger.info("\n模型选择器测试完成!")
        
    except ImportError as e:
        logger.error(f"导入失败: {e}")
        logger.info("请确保在正确的目录中运行此脚本")
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_selector() 