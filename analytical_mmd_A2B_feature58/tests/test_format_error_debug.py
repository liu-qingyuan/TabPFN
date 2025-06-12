#!/usr/bin/env python3
"""
调试格式化错误的测试脚本
"""

import sys
import os
import traceback

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

def test_format_error_debug():
    """调试格式化错误"""
    try:
        # 模拟改进幅度数据
        improvement_data = {
            'accuracy': '-0.0210',
            'auc': '-0.1373', 
            'f1': '+0.0155'
        }
        
        print("测试改进幅度格式化:")
        for metric, value in improvement_data.items():
            print(f"  {metric}: {value}")
        
        # 测试 summarize_cross_domain_results 函数
        from metrics.cross_domain_metrics import summarize_cross_domain_results
        
        # 模拟结果数据
        test_results = {
            'external_validation_B': {
                'without_domain_adaptation': {
                    'accuracy': 0.6789,
                    'auc': 0.6673,
                    'f1': 0.7749
                },
                'with_domain_adaptation': {
                    'accuracy': 0.6579,
                    'auc': 0.5300,
                    'f1': 0.7904
                },
                'improvement': improvement_data
            }
        }
        
        print("\n测试 summarize_cross_domain_results:")
        summary = summarize_cross_domain_results(test_results)
        print(summary)
        
        return True
        
    except Exception as e:
        print(f"错误: {e}")
        print("完整错误追踪:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_format_error_debug()
    sys.exit(0 if success else 1) 