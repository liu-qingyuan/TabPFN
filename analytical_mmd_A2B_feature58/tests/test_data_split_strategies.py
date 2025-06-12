#!/usr/bin/env python3
"""
测试数据划分策略功能
"""

import sys
import os
import argparse
import logging

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 现在可以正确导入模块
from utils.logging_setup import setup_logger

def create_test_args(data_split_strategy: str = 'two-way', use_bayesian_optimization: bool = False):
    """创建测试用的参数对象"""
    from argparse import Namespace
    
    args = Namespace(
        # 模型相关参数 - 使用auto模型进行测试
        model_type='auto',  # 改为auto模型
        model_preset=None,
        
        # 特征相关参数
        feature_type='best7',
        
        # MMD方法参数
        method='linear',
        compare_all=False,
        
        # MMD参数
        gamma=None,
        lr=None,
        n_epochs=None,
        batch_size=None,
        lambda_reg=None,
        n_components=None,
        
        # 改进的MMD参数
        no_staged_training=False,
        no_dynamic_gamma=False,
        gamma_search_values=None,
        use_preset=None,
        
        # Linear方法调试参数
        standardize_features=False,
        use_gradient_clipping=False,
        max_grad_norm=1.0,
        monitor_gradients=False,
        
        # 核PCA改进参数
        use_inverse_transform=False,
        no_standardize=False,
        
        # 模型特定参数 - 设置更快的参数
        max_time=10,  # 减少训练时间到10秒
        max_models=5,  # 减少模型数量
        n_estimators=None,
        device=None,
        
        # 输出参数
        output_dir=None,
        no_visualizations=True,  # 禁用可视化
        log_file=None,
        
        # 实验参数
        use_class_conditional=False,
        use_target_labels=False,
        use_threshold_optimizer=False,
        skip_cv_on_a=True,
        evaluation_mode='cv',
        random_seed=42,
        
        # 数据划分策略参数
        data_split_strategy=data_split_strategy,
        validation_split=0.7,
        use_bayesian_optimization=use_bayesian_optimization,
        bo_n_calls=10 if use_bayesian_optimization else None,  # 改为10次，满足scikit-optimize最小要求
        bo_random_state=42 if use_bayesian_optimization else None,
        
        # 其他可能需要的参数
        mode=None,
        debug=False,
        save_intermediate=False
    )
    
    return args

def test_two_way_split():
    """测试二分法数据划分"""
    print("=" * 60)
    print("测试二分法数据划分")
    print("=" * 60)
    
    # 创建测试参数
    args = create_test_args(data_split_strategy='two-way', use_bayesian_optimization=False)
    
    # 设置日志
    logger = setup_logger(log_level=logging.INFO, console_output=True)
    
    try:
        # 导入主脚本函数
        from scripts.run_analytical_mmd import run_cross_domain_experiment_mode
        
        logger.info("开始测试二分法数据划分...")
        run_cross_domain_experiment_mode(args, logger)
        logger.info("✓ 二分法测试完成")
        
    except Exception as e:
        import traceback
        logger.error(f"二分法测试失败: {e}")
        logger.error("完整错误追踪:")
        logger.error(traceback.format_exc())
        return False
    
    return True

def test_three_way_split():
    """测试三分法数据划分"""
    print("=" * 60)
    print("测试三分法数据划分")
    print("=" * 60)
    
    # 创建测试参数
    args = create_test_args(data_split_strategy='three-way', use_bayesian_optimization=False)
    
    # 设置日志
    logger = setup_logger(log_level=logging.INFO, console_output=True)
    
    try:
        # 导入主脚本函数
        from scripts.run_analytical_mmd import run_cross_domain_experiment_mode
        
        logger.info("开始测试三分法数据划分...")
        run_cross_domain_experiment_mode(args, logger)
        logger.info("✓ 三分法测试完成")
        
    except Exception as e:
        logger.error(f"三分法测试失败: {e}")
        return False
    
    return True

def test_bayesian_optimization():
    """测试贝叶斯优化功能"""
    print("=" * 60)
    print("测试贝叶斯优化功能")
    print("=" * 60)
    
    # 创建测试参数
    args = create_test_args(data_split_strategy='three-way', use_bayesian_optimization=True)
    
    # 设置日志
    logger = setup_logger(log_level=logging.INFO, console_output=True)
    
    # 首先检查贝叶斯优化模块是否存在
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    bayesian_file = os.path.join(project_root, "modeling", "bayesian_optimizer.py")
    
    logger.info(f"检查贝叶斯优化模块: {bayesian_file}")
    if not os.path.exists(bayesian_file):
        logger.error(f"✗ 贝叶斯优化文件不存在: {bayesian_file}")
        
        # 列出modeling目录的内容
        modeling_dir = os.path.join(project_root, "modeling")
        if os.path.exists(modeling_dir):
            files = os.listdir(modeling_dir)
            logger.error(f"modeling目录内容: {files}")
        else:
            logger.error("modeling目录不存在")
        
        logger.error("贝叶斯优化模块缺失，测试失败")
        return False
    else:
        logger.info(f"✓ 贝叶斯优化文件存在: {bayesian_file}")
    
    try:
        # 导入主脚本函数
        from scripts.run_analytical_mmd import run_bayesian_optimization_mode
        
        logger.info("开始测试贝叶斯优化...")
        logger.info(f"测试参数: data_split_strategy={args.data_split_strategy}, use_bayesian_optimization={args.use_bayesian_optimization}")
        logger.info(f"贝叶斯优化参数: bo_n_calls={args.bo_n_calls}, bo_random_state={args.bo_random_state}")
        
        # 创建一个自定义的日志处理器来捕获错误信息
        import io
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.ERROR)
        logger.addHandler(handler)
        
        run_bayesian_optimization_mode(args, logger)
        
        # 检查是否有错误信息
        log_contents = log_capture.getvalue()
        if "无法导入贝叶斯优化模块" in log_contents or "attempted relative import" in log_contents:
            logger.error("检测到贝叶斯优化模块导入失败")
            logger.error(f"错误日志: {log_contents}")
            return False
        
        logger.info("✓ 贝叶斯优化测试完成")
        
    except Exception as e:
        import traceback
        logger.error(f"贝叶斯优化测试失败: {e}")
        logger.error("完整错误追踪:")
        logger.error(traceback.format_exc())
        return False
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='测试数据划分策略功能')
    parser.add_argument('--test', type=str, choices=['two-way', 'three-way', 'bayesian', 'all'], 
                       default='all', help='要运行的测试')
    
    args = parser.parse_args()
    
    success_count = 0
    total_count = 0
    
    if args.test in ['two-way', 'all']:
        total_count += 1
        if test_two_way_split():
            success_count += 1
    
    if args.test in ['three-way', 'all']:
        total_count += 1
        if test_three_way_split():
            success_count += 1
    
    if args.test in ['bayesian', 'all']:
        total_count += 1
        if test_bayesian_optimization():
            success_count += 1
    
    print("=" * 60)
    print(f"测试结果: {success_count}/{total_count} 通过")
    print("=" * 60)
    
    if success_count == total_count:
        print("✓ 所有测试通过!")
        return 0
    else:
        print("✗ 部分测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 