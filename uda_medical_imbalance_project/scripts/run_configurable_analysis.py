#!/usr/bin/env python3
"""
基于配置文件的完整医疗数据UDA分析流程

这个脚本支持通过YAML配置文件进行灵活的参数配置，
提供完整的分析流程：
1. 源域10折交叉验证对比（TabPFN、论文方法、基线模型）
2. UDA域适应方法对比（基于ADAPT库）
3. 可视化分析和结果对比

运行示例: 
python scripts/run_configurable_analysis.py
python scripts/run_configurable_analysis.py --config configs/complete_analysis_config.yaml
"""

import sys
import os
import argparse
from pathlib import Path
import yaml
from typing import Dict, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.run_complete_analysis import CompleteAnalysisRunner


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def create_runner_from_config(config: Dict[str, Any]) -> CompleteAnalysisRunner:
    """根据配置创建分析运行器"""
    
    # 提取基本配置
    experiment_config = config.get('experiment', {})
    preprocessing_config = config.get('preprocessing', {})
    source_domain_config = config.get('source_domain', {})
    output_config = config.get('output', {})
    
    # 创建输出目录名称
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = experiment_config.get('name', 'complete_analysis')
    output_dir = f"{output_config.get('results_dir', 'results')}/{experiment_name}_{timestamp}"
    
    # 创建运行器
    runner = CompleteAnalysisRunner(
        feature_set=preprocessing_config.get('feature_set', 'best8'),
        scaler_type=preprocessing_config.get('scaler', 'standard'),
        imbalance_method=preprocessing_config.get('imbalance_method', 'smote'),
        cv_folds=source_domain_config.get('cv_folds', 10),
        random_state=experiment_config.get('random_state', 42),
        output_dir=output_dir,
        verbose=experiment_config.get('verbose', True)
    )
    
    # 将完整配置保存到runner中，以便后续使用
    runner.config = config
    
    return runner


def run_configurable_analysis(config_path: str):
    """运行基于配置文件的分析"""
    
    print(f"🔧 加载配置文件: {config_path}")
    
    try:
        # 加载配置
        config = load_config(config_path)
        
        # 显示配置信息
        experiment_config = config.get('experiment', {})
        print(f"📋 实验配置:")
        print(f"   名称: {experiment_config.get('name', 'N/A')}")
        print(f"   描述: {experiment_config.get('description', 'N/A')}")
        print(f"   随机种子: {experiment_config.get('random_state', 42)}")
        
        preprocessing_config = config.get('preprocessing', {})
        print(f"🔧 预处理配置:")
        print(f"   特征集: {preprocessing_config.get('feature_set', 'best8')}")
        print(f"   标准化: {preprocessing_config.get('scaler', 'standard')}")
        print(f"   不平衡处理: {preprocessing_config.get('imbalance_method', 'smote')}")
        
        source_domain_config = config.get('source_domain', {})
        print(f"📊 源域配置:")
        print(f"   交叉验证折数: {source_domain_config.get('cv_folds', 10)}")
        print(f"   模型列表: {source_domain_config.get('models', [])}")
        
        # 显示UDA方法配置
        uda_config = config.get('uda_methods', {})
        print(f"🔄 UDA方法配置:")
        
        instance_methods = uda_config.get('instance_based', [])
        if instance_methods:
            print(f"   实例重加权方法: {[m['method'] for m in instance_methods]}")
        
        feature_methods = uda_config.get('feature_based', [])
        if feature_methods:
            print(f"   特征对齐方法: {[m['method'] for m in feature_methods]}")
        
        print()
        
        # 创建分析运行器
        runner = create_runner_from_config(config)
        
        # 运行分析
        results = runner.run_complete_analysis()
        
        if 'error' not in results:
            print(f"\n🎉 配置化分析成功完成！")
            print(f"📁 查看结果目录: {runner.output_dir}")
            
            # 保存使用的配置到结果目录
            config_backup_path = runner.output_dir / "used_config.yaml"
            with open(config_backup_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            print(f"📋 使用的配置已备份: {config_backup_path}")
            
        else:
            print(f"\n❌ 配置化分析失败: {results['error']}")
            
    except Exception as e:
        print(f"❌ 配置化分析过程失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="基于配置文件的完整医疗数据UDA分析流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认配置文件
  python scripts/run_configurable_analysis.py
  
  # 使用指定配置文件
  python scripts/run_configurable_analysis.py --config configs/custom_config.yaml
  
  # 使用不同的特征集
  python scripts/run_configurable_analysis.py --feature-set best7
  
  # 使用不同的不平衡处理方法
  python scripts/run_configurable_analysis.py --imbalance-method borderline_smote
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/complete_analysis_config.yaml',
        help='配置文件路径 (默认: configs/complete_analysis_config.yaml)'
    )
    
    parser.add_argument(
        '--feature-set',
        type=str,
        choices=['best7', 'best8', 'best9', 'best10', 'all'],
        help='覆盖配置文件中的特征集设置'
    )
    
    parser.add_argument(
        '--scaler',
        type=str,
        choices=['standard', 'robust', 'none'],
        help='覆盖配置文件中的标准化设置'
    )
    
    parser.add_argument(
        '--imbalance-method',
        type=str,
        choices=['none', 'smote', 'smotenc', 'borderline_smote', 'kmeans_smote', 
                'svm_smote', 'adasyn', 'smote_tomek', 'smote_enn', 'random_under'],
        help='覆盖配置文件中的不平衡处理设置'
    )
    
    parser.add_argument(
        '--cv-folds',
        type=int,
        help='覆盖配置文件中的交叉验证折数设置'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        help='覆盖配置文件中的随机种子设置'
    )
    
    args = parser.parse_args()
    
    print("🏥 基于配置文件的完整医疗数据UDA分析")
    print("=" * 60)
    
    # 检查配置文件是否存在
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {args.config}")
        
        # 如果是默认配置文件不存在，提供创建建议
        if args.config == 'configs/complete_analysis_config.yaml':
            print("\n💡 建议:")
            print("1. 创建configs目录: mkdir -p configs")
            print("2. 复制示例配置: cp configs/complete_analysis_config.yaml configs/my_config.yaml")
            print("3. 编辑配置文件以满足你的需求")
            print("4. 重新运行: python scripts/run_configurable_analysis.py --config configs/my_config.yaml")
        
        return
    
    # 如果有命令行参数覆盖，先加载配置然后修改
    if any([args.feature_set, args.scaler, args.imbalance_method, args.cv_folds, args.random_state]):
        print(f"⚙️ 检测到命令行参数覆盖，正在修改配置...")
        
        # 加载原始配置
        config = load_config(args.config)
        
        # 应用命令行覆盖
        if args.feature_set:
            config.setdefault('preprocessing', {})['feature_set'] = args.feature_set
            print(f"   特征集覆盖为: {args.feature_set}")
            
        if args.scaler:
            config.setdefault('preprocessing', {})['scaler'] = args.scaler
            print(f"   标准化覆盖为: {args.scaler}")
            
        if args.imbalance_method:
            config.setdefault('preprocessing', {})['imbalance_method'] = args.imbalance_method
            print(f"   不平衡处理覆盖为: {args.imbalance_method}")
            
        if args.cv_folds:
            config.setdefault('source_domain', {})['cv_folds'] = args.cv_folds
            print(f"   交叉验证折数覆盖为: {args.cv_folds}")
            
        if args.random_state:
            config.setdefault('experiment', {})['random_state'] = args.random_state
            print(f"   随机种子覆盖为: {args.random_state}")
        
        # 保存修改后的配置到临时文件
        temp_config_path = Path("temp_config.yaml")
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        # 使用临时配置文件
        config_path = temp_config_path
        print()
    
    # 运行分析
    try:
        run_configurable_analysis(str(config_path))
    finally:
        # 清理临时配置文件
        if config_path.name == "temp_config.yaml" and config_path.exists():
            config_path.unlink()


if __name__ == "__main__":
    main() 