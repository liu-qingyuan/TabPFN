#!/usr/bin/env python3
"""
CSV数据读取测试脚本
验证数据文件格式和内容的简化版本
"""

import csv
from pathlib import Path

def test_csv_data():
    """测试CSV数据文件"""
    csv_path = Path("/Users/lqy/work/TabPFN/uda_medical_imbalance_project/results/feature_number_evaluation/feature_number_comparison.csv")
    
    print(f"🔍 测试CSV文件: {csv_path}")
    
    if not csv_path.exists():
        print(f"❌ 文件不存在: {csv_path}")
        return False
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            
            print(f"✅ CSV文件可读取")
            if headers:
                print(f"📋 列名: {list(headers)}")
            
            rows = list(reader)
            print(f"📊 数据行数: {len(rows)}")
            
            if len(rows) > 0:
                print(f"📈 特征数量范围: {rows[0]['n_features']} - {rows[-1]['n_features']}")
                
                # 检查关键列
                required_cols = ['n_features', 'mean_accuracy', 'mean_auc', 'mean_f1', 'mean_time']
                missing_cols = [col for col in required_cols if col not in (headers or [])]
                
                if missing_cols:
                    print(f"⚠️ 缺少必要列: {missing_cols}")
                else:
                    print("✅ 包含所有必要的数据列")
                
                # 显示前几行数据
                print(f"\n📋 前3行数据预览:")
                for row in rows[:3]:
                    features = row['n_features']
                    accuracy = row['mean_accuracy']
                    auc = row['mean_auc']
                    f1 = row['mean_f1']
                    time = row['mean_time']
                    print(f"  特征数: {features}, 准确率: {accuracy}, AUC: {auc}, F1: {f1}, 时间: {time}")
                
                return True
            else:
                print("❌ 文件为空")
                return False
                
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return False

if __name__ == "__main__":
    print("🔍 CSV数据文件测试")
    print("=" * 40)
    
    success = test_csv_data()
    
    if success:
        print("\n✅ CSV文件测试通过！")
        print("💡 可以运行完整的PDF生成脚本:")
        print("   python generate_feature_performance_pdf.py")
    else:
        print("\n❌ CSV文件测试失败")
        print("请检查数据文件是否存在且格式正确")