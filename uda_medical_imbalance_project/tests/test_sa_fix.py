#!/usr/bin/env python3
"""测试修复后的SA方法"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_adapt_methods import load_test_data, create_tabpfn_model, calculate_metrics
from uda.adapt_methods import create_adapt_method, is_adapt_available
import numpy as np

def test_sa_fix():
    if not is_adapt_available():
        print('Adapt库不可用')
        return
        
    X_A, y_A, X_B, y_B = load_test_data()
    print('=== 修复后的SA方法测试 ===')
    
    # 基线性能
    baseline_model = create_tabpfn_model()
    baseline_model.fit(X_A, y_A)
    baseline_pred = baseline_model.predict(X_B)
    baseline_proba = baseline_model.predict_proba(X_B)
    baseline_metrics = calculate_metrics(y_B, baseline_pred, baseline_proba)
    
    print('基线TabPFN性能:')
    for metric, value in baseline_metrics.items():
        if not np.isnan(value):
            print(f'  {metric.upper()}: {value:.4f}')
    
    # 创建SA方法
    sa_method = create_adapt_method('SA', estimator=create_tabpfn_model(), random_state=42)
    sa_method.fit(X_A.values, y_A.values, X_B.values)
    
    # 预测
    sa_pred = sa_method.predict(X_B.values)
    sa_proba = sa_method.predict_proba(X_B.values)
    
    if sa_proba is not None:
        print(f'\nSA概率形状: {sa_proba.shape}')
        print(f'SA概率[:, 1]范围: [{sa_proba[:, 1].min():.4f}, {sa_proba[:, 1].max():.4f}]')
        
        # 计算指标
        metrics = calculate_metrics(y_B, sa_pred, sa_proba)
        print(f'\n修复后SA性能:')
        for metric, value in metrics.items():
            if not np.isnan(value):
                print(f'  {metric.upper()}: {value:.4f}')
                
        # 对比改进
        print(f'\n性能对比（SA vs 基线）:')
        for metric in ['accuracy', 'auc', 'f1', 'precision', 'recall']:
            if metric in metrics and metric in baseline_metrics:
                baseline_val = baseline_metrics[metric]
                sa_val = metrics[metric]
                if not np.isnan(baseline_val) and not np.isnan(sa_val):
                    improvement = sa_val - baseline_val
                    print(f'  {metric.upper()}: {baseline_val:.4f} → {sa_val:.4f} '
                          f'({"+"if improvement >= 0 else ""}{improvement:.4f})')
    else:
        print('SA概率预测仍然失败')

if __name__ == "__main__":
    test_sa_fix() 