#!/usr/bin/env python3
"""
完整医疗数据UDA分析流程使用示例

这个示例展示如何使用完整的分析流程：
1. 源域10折交叉验证对比（TabPFN、论文方法、基线模型）
2. UDA域适应方法对比（基于ADAPT库）
3. 可视化分析和结果对比

运行示例: python examples/complete_analysis_example.py
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def example_basic_analysis():
    """基本分析示例"""
    
    print("📋 示例1: 基本完整分析")
    print("=" * 50)
    
    from scripts.run_complete_analysis import CompleteAnalysisRunner
    
    # 创建分析运行器
    runner = CompleteAnalysisRunner(
        feature_type='best8',        # 使用best8特征类型
        scaler_type='standard',     # 标准化
        imbalance_method='smote',   # SMOTE处理不平衡
        cv_folds=10,               # 10折交叉验证
        random_state=42,           # 固定随机种子
        verbose=True               # 显示详细信息
    )
    
    # 运行完整分析
    results = runner.run_complete_analysis()
    
    if 'error' not in results:
        print(f"✅ 基本分析完成！结果保存在: {runner.output_dir}")
        return True
    else:
        print(f"❌ 基本分析失败: {results['error']}")
        return False


def example_custom_parameters():
    """自定义参数分析示例"""
    
    print("\n📋 示例2: 自定义参数分析")
    print("=" * 50)
    
    from scripts.run_complete_analysis import CompleteAnalysisRunner
    
    # 创建自定义配置的分析运行器
    runner = CompleteAnalysisRunner(
        feature_type='best7',              # 使用best7特征类型
        scaler_type='robust',             # 鲁棒标准化
        imbalance_method='borderline_smote',  # BorderlineSMOTE
        cv_folds=5,                       # 5折交叉验证（更快）
        random_state=123,                 # 不同的随机种子
        output_dir='results/custom_analysis',  # 自定义输出目录
        verbose=True
    )
    
    # 运行分析
    results = runner.run_complete_analysis()
    
    if 'error' not in results:
        print(f"✅ 自定义参数分析完成！结果保存在: {runner.output_dir}")
        return True
    else:
        print(f"❌ 自定义参数分析失败: {results['error']}")
        return False


def example_config_based_analysis():
    """基于配置文件的分析示例"""
    
    print("\n📋 示例3: 基于配置文件的分析")
    print("=" * 50)
    
    try:
        from scripts.run_configurable_analysis import run_configurable_analysis
        
        # 使用配置文件运行分析
        config_path = "configs/complete_analysis_config.yaml"
        
        # 检查配置文件是否存在
        if not Path(config_path).exists():
            print(f"⚠ 配置文件不存在: {config_path}")
            print("请先创建配置文件或使用其他示例")
            return False
        
        # 运行配置化分析
        run_configurable_analysis(config_path)
        
        print(f"✅ 配置化分析完成！")
        return True
        
    except Exception as e:
        print(f"❌ 配置化分析失败: {e}")
        return False


def example_step_by_step_analysis():
    """分步骤分析示例"""
    
    print("\n📋 示例4: 分步骤分析")
    print("=" * 50)
    
    try:
        from scripts.run_complete_analysis import CompleteAnalysisRunner
        
        # 创建分析运行器
        runner = CompleteAnalysisRunner(
            feature_type='best8',
            scaler_type='standard',
            imbalance_method='smote',
            cv_folds=3,  # 减少折数以加快演示
            random_state=42,
            output_dir='results/step_by_step_analysis',
            verbose=True
        )
        
        print("📊 步骤1: 加载数据...")
        X_source, y_source, X_target, y_target, feature_names = runner.load_data()
        print(f"✅ 数据加载完成: 源域{X_source.shape}, 目标域{X_target.shape}")
        
        print("\n🔬 步骤2: 源域交叉验证...")
        cv_results = runner.run_source_domain_cv(X_source, y_source)
        print(f"✅ 源域CV完成: {len(cv_results)}个实验")
        
        print("\n🔄 步骤3: UDA方法对比...")
        uda_results = runner.run_uda_methods(X_source, y_source, X_target, y_target, feature_names)
        successful_uda = len([k for k, v in uda_results.items() if 'error' not in v])
        print(f"✅ UDA方法完成: {successful_uda}个方法成功")
        
        print("\n📊 步骤4: 生成可视化...")
        viz_results = runner.generate_comparison_visualizations()
        print(f"✅ 可视化完成: {len(viz_results)}个图表")
        
        print("\n📋 步骤5: 生成报告...")
        report_file = runner.generate_final_report()
        print(f"✅ 报告完成: {report_file}")
        
        print(f"\n🎉 分步骤分析完成！结果保存在: {runner.output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ 分步骤分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def example_comparison_analysis():
    """对比分析示例：不同配置的对比"""
    
    print("\n📋 示例5: 对比分析")
    print("=" * 50)
    
    from scripts.run_complete_analysis import CompleteAnalysisRunner
    
    # 定义不同的配置
    configs = [
        {
            'name': 'Config1_SMOTE',
            'feature_set': 'best8',
            'scaler_type': 'standard',
            'imbalance_method': 'smote'
        },
        {
            'name': 'Config2_BorderlineSMOTE', 
            'feature_set': 'best8',
            'scaler_type': 'standard',
            'imbalance_method': 'borderline_smote'
        },
        {
            'name': 'Config3_NoImbalanceHandling',
            'feature_set': 'best8', 
            'scaler_type': 'standard',
            'imbalance_method': 'none'
        }
    ]
    
    comparison_results = {}
    
    for config in configs:
        print(f"\n🔧 运行配置: {config['name']}")
        
        try:
            runner = CompleteAnalysisRunner(
                feature_type=config['feature_set'],
                scaler_type=config['scaler_type'],
                imbalance_method=config['imbalance_method'],
                cv_folds=3,  # 减少折数以加快对比
                random_state=42,
                output_dir=f"results/comparison_{config['name']}",
                verbose=False  # 减少输出
            )
            
            results = runner.run_complete_analysis()
            
            if 'error' not in results:
                comparison_results[config['name']] = results
                print(f"✅ {config['name']} 完成")
            else:
                print(f"❌ {config['name']} 失败: {results['error']}")
                
        except Exception as e:
            print(f"❌ {config['name']} 异常: {e}")
    
    # 输出对比结果
    if comparison_results:
        print(f"\n📊 对比结果总结:")
        print("=" * 60)
        
        for config_name, results in comparison_results.items():
            print(f"\n{config_name}:")
            
            # 源域最佳结果
            if 'source_domain_cv' in results:
                cv_results = results['source_domain_cv']
                best_auc = 0
                best_method = ""
                
                for exp_name, result in cv_results.items():
                    if 'summary' in result and result['summary']:
                        auc = result['summary'].get('auc_mean', 0)
                        if auc > best_auc:
                            best_auc = auc
                            best_method = exp_name.split('_')[0]
                
                print(f"  源域最佳: {best_method} (AUC: {best_auc:.4f})")
            
            # UDA最佳结果
            if 'uda_methods' in results:
                uda_results = results['uda_methods']
                successful_uda = {k: v for k, v in uda_results.items() if 'error' not in v}
                
                if successful_uda:
                    best_uda_auc = 0
                    best_uda_method = ""
                    
                    for method, result in successful_uda.items():
                        auc = result.get('auc', 0) if result.get('auc') is not None else 0
                        if auc > best_uda_auc:
                            best_uda_auc = auc
                            best_uda_method = method
                    
                    print(f"  UDA最佳: {best_uda_method} (AUC: {best_uda_auc:.4f})")
                else:
                    print(f"  UDA最佳: 无成功方法")
        
        print(f"\n✅ 对比分析完成！")
        return True
    else:
        print(f"❌ 所有配置都失败了")
        return False


def main():
    """主函数：运行所有示例"""
    
    print("🏥 完整医疗数据UDA分析流程示例")
    print("=" * 60)
    
    examples = [
        ("基本分析", example_basic_analysis),
        ("自定义参数分析", example_custom_parameters),
        ("配置文件分析", example_config_based_analysis),
        ("分步骤分析", example_step_by_step_analysis),
        ("对比分析", example_comparison_analysis)
    ]
    
    success_count = 0
    
    for i, (name, func) in enumerate(examples, 1):
        print(f"\n{'='*20} 示例 {i}/{len(examples)}: {name} {'='*20}")
        
        try:
            if func():
                success_count += 1
                print(f"✅ 示例 {i} 成功")
            else:
                print(f"❌ 示例 {i} 失败")
        except KeyboardInterrupt:
            print(f"\n⏹ 用户中断")
            break
        except Exception as e:
            print(f"❌ 示例 {i} 异常: {e}")
        
        # 询问是否继续下一个示例
        if i < len(examples):
            try:
                response = input(f"\n继续下一个示例？(y/n, 默认y): ").strip().lower()
                if response in ['n', 'no']:
                    break
            except KeyboardInterrupt:
                print(f"\n⏹ 用户中断")
                break
    
    # 总结
    print(f"\n" + "=" * 60)
    print(f"示例运行总结: {success_count}/{len(examples)} 成功")
    
    if success_count > 0:
        print(f"🎉 至少有 {success_count} 个示例成功运行！")
        print(f"💡 建议:")
        print(f"   1. 查看 results/ 目录下的分析结果")
        print(f"   2. 根据需要修改配置参数")
        print(f"   3. 使用真实数据运行完整分析")
    else:
        print(f"❌ 所有示例都失败了，请检查环境配置")
    
    print(f"=" * 60)


if __name__ == "__main__":
    main() 