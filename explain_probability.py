import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, confusion_matrix
from tabpfn import TabPFNClassifier

print("=" * 60)
print("解释TabPFN预测概率和AUC")
print("=" * 60)

# 读取数据
df = pd.read_excel('data/HenanCancerHospital_translated_english.xlsx')
features = [c for c in df.columns if c.startswith('Feature')]
X = df[features]
y = df['Label']

print(f"数据概况:")
print(f"总样本数: {len(df)}")
print(f"阳性样本(癌症): {y.sum()}")
print(f"阴性样本(健康): {len(y) - y.sum()}")
print(f"阳性比例: {y.mean():.1%}")

# 简单的交叉验证示例
kf = KFold(n_splits=2, shuffle=True, random_state=42)
train_idx, test_idx = next(kf.split(X))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print(f"\n训练集阳性比例: {y_train.mean():.1%}")
print(f"测试集阳性比例: {y_test.mean():.1%}")

# 训练模型
clf = TabPFNClassifier(device='cuda', n_estimators=32, random_state=42)
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)

print("\n" + "=" * 60)
print("1. 预测概率解释")
print("=" * 60)

print('预测概率示例（前15个样本）:')
print('索引 | 真实 | 预测 | 阴性概率 | 阳性概率')
print('-' * 45)
for i in range(min(15, len(y_test))):
    real_label = y_test.iloc[i]
    pred_label = y_pred[i]
    neg_prob = y_pred_proba[i, 0]
    pos_prob = y_pred_proba[i, 1]
    marker = "✓" if real_label == pred_label else "✗"
    print(f'{test_idx[i]:4d} | {real_label:4d} | {pred_label:4d} | {neg_prob:7.3f} | {pos_prob:7.3f} {marker}')

# 计算平均恶性概率
all_avg_pos_prob = y_pred_proba[:, 1].mean()
cancer_samples_mask = y_test.values == 1
healthy_samples_mask = y_test.values == 0

if cancer_samples_mask.sum() > 0:
    cancer_avg_pos_prob = y_pred_proba[cancer_samples_mask, 1].mean()
else:
    cancer_avg_pos_prob = 0

if healthy_samples_mask.sum() > 0:
    healthy_avg_pos_prob = y_pred_proba[healthy_samples_mask, 1].mean()
else:
    healthy_avg_pos_prob = 0

print(f'\n所有样本的平均阳性概率: {all_avg_pos_prob:.3f}')
print(f'真实癌症样本的平均阳性概率: {cancer_avg_pos_prob:.3f}')
print(f'真实健康样本的平均阳性概率: {healthy_avg_pos_prob:.3f}')

print("\n💡 平均恶性概率计算方式:")
print("   对于某种癌症类型的所有样本，将它们的预测阳性概率相加后除以样本数")
print("   例如：Adenocarcinoma的99个样本，每个样本都有一个0-1之间的预测概率")
print("   如果这99个概率加起来是67.6，那么平均概率就是 67.6/99 ≈ 0.683")

print("\n" + "=" * 60)
print("2. AUC低的原因分析")
print("=" * 60)

# 计算AUC
auc = roc_auc_score(y_test, y_pred_proba[:, 1])
print(f"当前AUC: {auc:.3f}")

# 混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()

print(f"\n混淆矩阵:")
print(f"真阴性(TN): {tn:3d} | 假阳性(FP): {fp:3d}")
print(f"假阴性(FN): {fn:3d} | 真阳性(TP): {tp:3d}")

# 计算各种指标
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0

print(f"\n敏感度(Sensitivity): {sensitivity:.3f} - 正确识别癌症的比例")
print(f"特异度(Specificity): {specificity:.3f} - 正确识别健康的比例")
print(f"精确度(Precision): {precision:.3f} - 预测为癌症中真的是癌症的比例")

print("\n🔍 AUC低的可能原因:")
print("1. 类别不平衡: 数据中癌症样本比健康样本多很多")
print(f"   - 癌症样本: {y.sum()} ({y.mean():.1%})")
print(f"   - 健康样本: {len(y) - y.sum()} ({(1-y.mean()):.1%})")

print("2. 模型预测偏向性:")
if fp > tn:
    print("   - 模型倾向于预测为阳性(癌症)")
    print("   - 很多健康样本被错误预测为癌症")
elif fn > tp:
    print("   - 模型倾向于预测为阴性(健康)")
    print("   - 很多癌症样本被错误预测为健康")

print("3. 特征区分能力:")
print("   - 如果健康和癌症样本的特征分布重叠较多")
print("   - 模型难以准确区分两类样本")

# 分析预测概率分布
print(f"\n预测概率分布分析:")
print(f"阳性概率 < 0.3 的样本: {(y_pred_proba[:, 1] < 0.3).sum()}")
print(f"阳性概率 0.3-0.7 的样本: {((y_pred_proba[:, 1] >= 0.3) & (y_pred_proba[:, 1] < 0.7)).sum()}")
print(f"阳性概率 > 0.7 的样本: {(y_pred_proba[:, 1] >= 0.7).sum()}")

print("\n💡 提高AUC的可能方法:")
print("1. 数据平衡: 使用采样技术平衡正负样本")
print("2. 特征工程: 选择更有区分力的特征")
print("3. 模型调优: 调整TabPFN的参数")
print("4. 阈值优化: 找到最佳的分类阈值")

print("\n" + "=" * 60) 