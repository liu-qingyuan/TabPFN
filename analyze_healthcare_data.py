import pandas as pd
import numpy as np
from evidently.future.datasets import Dataset, DataDefinition
from evidently.future.report import Report
from evidently.future.presets import DataQualityPreset, DataDriftPreset, TargetDriftPreset
from evidently.future.tests import DataQualityTestPreset, DataDriftTestPreset
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """加载并准备数据集"""
    print("正在加载数据集...")
    
    # 加载三个数据集
    ai4health = pd.read_excel("data/AI4healthcare.xlsx")
    guangzhou = pd.read_excel("data/GuangzhouMedicalHospital_features23.xlsx")
    henan = pd.read_excel("data/HenanCancerHospital_features63_58.xlsx")
    
    # 获取所有数据集共有的特征
    common_features = set(ai4health.columns)
    common_features = common_features.intersection(set(guangzhou.columns))
    common_features = common_features.intersection(set(henan.columns))
    common_features = list(common_features - {'Label'})  # 移除标签列
    
    print(f"\n共同特征数量: {len(common_features)}")
    print("共同特征列表:", common_features)
    
    return ai4health, guangzhou, henan, common_features

def create_evidently_dataset(df, common_features):
    """创建Evidently数据集"""
    schema = DataDefinition(
        numerical_features=common_features,  # 所有特征都作为数值特征
        target='Label'  # 指定目标变量
    )
    
    return Dataset.from_pandas(df[common_features + ['Label']], data_definition=schema)

def analyze_datasets(dataset1, dataset2, name1, name2):
    """分析两个数据集之间的差异"""
    print(f"\n正在分析 {name1} vs {name2}...")
    
    # 创建包含多个预设的报告
    report = Report(metrics=[
        DataQualityPreset(),  # 数据质量分析
        DataDriftPreset(),    # 数据漂移分析
        TargetDriftPreset()   # 目标变量漂移分析
    ])
    
    # 运行分析
    analysis = report.run(reference_data=dataset1, current_data=dataset2)
    
    # 保存HTML报告
    report_name = f"results/data_analysis_{name1}_vs_{name2}.html"
    analysis.save_html(report_name)
    print(f"分析报告已保存至: {report_name}")
    
    # 运行测试套件
    test_suite = TestSuite(tests=[
        DataQualityTestPreset(),  # 数据质量测试
        DataDriftTestPreset()     # 数据漂移测试
    ])
    
    # 运行测试
    test_results = test_suite.run(reference_data=dataset1, current_data=dataset2)
    
    # 保存测试结果
    test_results_name = f"results/test_results_{name1}_vs_{name2}.html"
    test_results.save_html(test_results_name)
    print(f"测试结果已保存至: {test_results_name}")
    
    return analysis, test_results

def main():
    """主函数"""
    # 创建results目录
    import os
    os.makedirs("results", exist_ok=True)
    
    # 加载数据
    ai4health, guangzhou, henan, common_features = load_and_prepare_data()
    
    # 创建Evidently数据集
    ai4health_dataset = create_evidently_dataset(ai4health, common_features)
    guangzhou_dataset = create_evidently_dataset(guangzhou, common_features)
    henan_dataset = create_evidently_dataset(henan, common_features)
    
    # 进行数据集对比分析
    analyze_datasets(ai4health_dataset, guangzhou_dataset, "AI4Health", "Guangzhou")
    analyze_datasets(ai4health_dataset, henan_dataset, "AI4Health", "Henan")
    analyze_datasets(guangzhou_dataset, henan_dataset, "Guangzhou", "Henan")
    
    print("\n分析完成！请查看results目录下的HTML报告文件获取详细分析结果。")

if __name__ == "__main__":
    main() 