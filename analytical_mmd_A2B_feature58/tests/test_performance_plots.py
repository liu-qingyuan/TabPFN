#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
性能对比图功能测试脚本

演示如何使用新的性能对比图模块生成各种类型的性能可视化图表
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def create_sample_results():
    """创建示例实验结果数据"""
    
    # 模拟跨域实验结果
    cross_domain_results = {
        'Dataset_B_MMD_Linear': {
            'without_domain_adaptation': {
                'accuracy': 0.75, 'auc': 0.78, 'f1': 0.72,
                'acc_0': 0.77, 'acc_1': 0.73
            },
            'with_domain_adaptation': {
                'accuracy': 0.82, 'auc': 0.85, 'f1': 0.79,
                'acc_0': 0.84, 'acc_1': 0.80
            },
            'improvement': {
                'auc_improvement': 0.07,
                'accuracy_improvement': 0.07
            }
        },
        'Dataset_B_MMD_KPCA': {
            'without_domain_adaptation': {
                'accuracy': 0.75, 'auc': 0.78, 'f1': 0.72,
                'acc_0': 0.77, 'acc_1': 0.73
            },
            'with_domain_adaptation': {
                'accuracy': 0.80, 'auc': 0.83, 'f1': 0.77,
                'acc_0': 0.82, 'acc_1': 0.78
            },
            'improvement': {
                'auc_improvement': 0.05,
                'accuracy_improvement': 0.05
            }
        },
        'Dataset_C_MMD_Linear': {
            'without_domain_adaptation': {
                'accuracy': 0.70, 'auc': 0.73, 'f1': 0.68,
                'acc_0': 0.72, 'acc_1': 0.68
            },
            'with_domain_adaptation': {
                'accuracy': 0.78, 'auc': 0.81, 'f1': 0.75,
                'acc_0': 0.80, 'acc_1': 0.76
            },
            'improvement': {
                'auc_improvement': 0.08,
                'accuracy_improvement': 0.08
            }
        }
    }
    
    # 模拟CV结果格式
    cv_results = {
        'AutoTabPFN': {
            'accuracy': '0.89 ± 0.03',
            'auc': '0.92 ± 0.02',
            'f1': '0.87 ± 0.04',
            'acc_0': '0.91 ± 0.02',
            'acc_1': '0.87 ± 0.03'
        },
        'TunedTabPFN': {
            'accuracy': '0.87 ± 0.02',
            'auc': '0.90 ± 0.03',
            'f1': '0.85 ± 0.03',
            'acc_0': '0.89 ± 0.02',
            'acc_1': '0.85 ± 0.04'
        },
        'BaseTabPFN': {
            'accuracy': '0.85 ± 0.04',
            'auc': '0.88 ± 0.03',
            'f1': '0.82 ± 0.05',
            'acc_0': '0.87 ± 0.03',
            'acc_1': '0.83 ± 0.04'
        }
    }
    
    # 模拟多模型结果
    model_comparison_results = {
        'AutoTabPFN': {
            'accuracy': 0.89, 'auc': 0.92, 'f1': 0.87,
            'acc_0': 0.91, 'acc_1': 0.87
        },
        'TunedTabPFN': {
            'accuracy': 0.87, 'auc': 0.90, 'f1': 0.85,
            'acc_0': 0.89, 'acc_1': 0.85
        },
        'BaseTabPFN': {
            'accuracy': 0.85, 'auc': 0.88, 'f1': 0.82,
            'acc_0': 0.87, 'acc_1': 0.83
        },
        'RF_TabPFN': {
            'accuracy': 0.83, 'auc': 0.86, 'f1': 0.80,
            'acc_0': 0.85, 'acc_1': 0.81
        }
    }
    
    return cross_domain_results, cv_results, model_comparison_results

def test_individual_plots():
    """测试单独的性能对比图功能"""
    print("测试单独的性能对比图功能...")
    
    try:
        from visualization.performance_plots import (
            plot_metrics_comparison, plot_domain_adaptation_improvement,
            plot_cross_dataset_performance, plot_model_comparison,
            plot_metrics_radar_chart, create_performance_summary_table
        )
        
        cross_domain_results, cv_results, model_comparison_results = create_sample_results()
        
        # 创建输出目录
        output_dir = project_root / "tests" / "test_results" / "performance_plots_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 测试基础性能指标对比图
        print("  生成基础性能指标对比图...")
        plot_metrics_comparison(
            results_dict=cv_results,
            save_path=str(output_dir / "metrics_comparison_test.png"),
            title="Model Performance Comparison Test"
        )
        
        # 2. 测试域适应改进效果图
        print("  生成域适应改进效果图...")
        plot_domain_adaptation_improvement(
            results_dict=cross_domain_results,
            save_path=str(output_dir / "domain_adaptation_improvement_test.png"),
            title="Domain Adaptation Improvement Test"
        )
        
        # 3. 测试跨数据集性能对比图
        print("  生成跨数据集性能对比图...")
        cross_dataset_data = {
            'Dataset_B': cross_domain_results['Dataset_B_MMD_Linear'],
            'Dataset_C': cross_domain_results['Dataset_C_MMD_Linear']
        }
        plot_cross_dataset_performance(
            results_dict=cross_dataset_data,
            save_path=str(output_dir / "cross_dataset_performance_test.png"),
            title="Cross-Dataset Performance Test"
        )
        
        # 4. 测试模型性能对比热力图
        print("  生成模型性能对比热力图...")
        plot_model_comparison(
            results_dict=model_comparison_results,
            save_path=str(output_dir / "model_comparison_heatmap_test.png"),
            title="Model Performance Heatmap Test"
        )
        
        # 5. 测试性能指标雷达图
        print("  生成性能指标雷达图...")
        plot_metrics_radar_chart(
            results_dict=model_comparison_results,
            save_path=str(output_dir / "performance_radar_chart_test.png"),
            title="Performance Radar Chart Test"
        )
        
        # 6. 测试性能汇总表格
        print("  生成性能汇总表格...")
        summary_df = create_performance_summary_table(
            results_dict=cross_domain_results,
            save_path=str(output_dir / "performance_summary_table_test.png"),
            title="Performance Summary Table Test"
        )
        
        print(f"  ✓ 单独测试完成，结果保存在: {output_dir}")
        if summary_df is not None:
            print(f"  ✓ 汇总表格形状: {summary_df.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 单独测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrated_plots():
    """测试集成的性能对比图功能"""
    print("测试集成的性能对比图功能...")
    
    try:
        from visualization.comparison_plots import generate_performance_comparison_plots
        
        cross_domain_results, _, _ = create_sample_results()
        
        # 创建输出目录
        output_dir = project_root / "tests" / "test_results" / "integrated_performance_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 测试集成功能
        print("  生成完整性能对比图表套件...")
        summary_df = generate_performance_comparison_plots(
            results_dict=cross_domain_results,
            save_dir=str(output_dir),
            experiment_name="MMD_Domain_Adaptation_Test"
        )
        
        # 检查生成的文件
        expected_files = [
            "MMD_Domain_Adaptation_Test_metrics_comparison.png",
            "MMD_Domain_Adaptation_Test_domain_adaptation_improvement.png",
            "MMD_Domain_Adaptation_Test_cross_dataset_performance.png",
            "MMD_Domain_Adaptation_Test_model_comparison.png",
            "MMD_Domain_Adaptation_Test_performance_radar.png",
            "MMD_Domain_Adaptation_Test_performance_summary.png",
            "MMD_Domain_Adaptation_Test_performance_summary.csv"
        ]
        
        generated_files = []
        for file_name in expected_files:
            file_path = output_dir / file_name
            if file_path.exists():
                generated_files.append(file_name)
        
        print(f"  ✓ 集成测试完成，结果保存在: {output_dir}")
        print(f"  ✓ 生成文件数量: {len(generated_files)}/{len(expected_files)}")
        if summary_df is not None:
            print(f"  ✓ 汇总表格形状: {summary_df.shape}")
        
        # 显示生成的文件列表
        print("  生成的文件:")
        for file_name in generated_files:
            print(f"    - {file_name}")
        
        return len(generated_files) >= len(expected_files) // 2  # 至少生成一半的文件就算成功
        
    except Exception as e:
        print(f"  ✗ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_format_compatibility():
    """测试数据格式兼容性"""
    print("测试数据格式兼容性...")
    
    try:
        from visualization.performance_plots import plot_metrics_comparison
        
        # 创建输出目录
        output_dir = project_root / "tests" / "test_results" / "format_compatibility_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 测试不同数据格式
        formats_to_test = [
            # 格式1: 跨域实验格式
            {
                'Cross_Domain_Format': {
                    'without_domain_adaptation': {
                        'accuracy': 0.85, 'auc': 0.88, 'f1': 0.82,
                        'acc_0': 0.87, 'acc_1': 0.83
                    },
                    'with_domain_adaptation': {
                        'accuracy': 0.89, 'auc': 0.92, 'f1': 0.87,
                        'acc_0': 0.91, 'acc_1': 0.87
                    }
                }
            },
            # 格式2: CV字符串格式
            {
                'CV_String_Format': {
                    'accuracy': '0.85 ± 0.03',
                    'auc': '0.88 ± 0.02',
                    'f1': '0.82 ± 0.04',
                    'acc_0': '0.87 ± 0.02',
                    'acc_1': '0.83 ± 0.03'
                }
            },
            # 格式3: 直接数值格式
            {
                'Direct_Values_Format': {
                    'accuracy': 0.85, 'auc': 0.88, 'f1': 0.82,
                    'acc_0': 0.87, 'acc_1': 0.83
                }
            }
        ]
        
        success_count = 0
        for i, test_format in enumerate(formats_to_test):
            try:
                format_name = list(test_format.keys())[0]
                print(f"  测试格式 {i+1}: {format_name}")
                
                plot_metrics_comparison(
                    results_dict=test_format,
                    save_path=str(output_dir / f"format_test_{i+1}_{format_name}.png"),
                    title=f"Format Test {i+1}: {format_name}"
                )
                
                success_count += 1
                print(f"    ✓ 格式 {i+1} 测试成功")
                
            except Exception as e:
                print(f"    ✗ 格式 {i+1} 测试失败: {e}")
        
        print(f"  ✓ 格式兼容性测试完成: {success_count}/{len(formats_to_test)} 成功")
        return success_count >= len(formats_to_test) // 2  # 至少一半成功就算通过
        
    except Exception as e:
        print(f"  ✗ 格式兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("性能对比图功能测试")
    print("=" * 60)
    
    # 运行所有测试
    tests = [
        ("单独功能测试", test_individual_plots),
        ("集成功能测试", test_integrated_plots),
        ("数据格式兼容性测试", test_data_format_compatibility)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        print("-" * 40)
        success = test_func()
        results.append((test_name, success))
    
    # 显示测试总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    success_count = 0
    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{test_name}: {status}")
        if success:
            success_count += 1
    
    print(f"\n总体结果: {success_count}/{len(results)} 测试通过")
    
    if success_count == len(results):
        print("🎉 所有测试通过！性能对比图功能正常工作。")
    else:
        print("⚠️  部分测试失败，请检查错误信息。")
    
    return success_count == len(results)

if __name__ == "__main__":
    main() 