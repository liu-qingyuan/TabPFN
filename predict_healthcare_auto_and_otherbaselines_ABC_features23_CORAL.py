import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import logging
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 设置日志系统
# 先移除 root logger 里所有的 handler
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# 创建两个 handler：
stdout_handler = logging.StreamHandler(sys.stdout)  # 处理 INFO 及以上的日志
stderr_handler = logging.StreamHandler(sys.stderr)  # 处理 WARNING 及以上的日志

# 设置不同的日志级别：
stdout_handler.setLevel(logging.INFO)   # 只处理 INFO及以上
stderr_handler.setLevel(logging.WARNING)  # 只处理 WARNING 及以上

# 设置日志格式
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stdout_handler.setFormatter(formatter)
stderr_handler.setFormatter(formatter)

# 把 handler 添加到 root logger
logging.root.addHandler(stdout_handler)
logging.root.addHandler(stderr_handler)
logging.root.setLevel(logging.INFO)  # 让 root logger 处理 INFO 及以上的日志

# 定义PKUPH和Mayo模型
class PKUPHModel:
    """
    PKUPH模型的实现
    P(malignant) = e^x / (1+e^x)
    x = -4.496 + (0.07 × Feature2) + (0.676 × Feature48) + (0.736 × Feature49) + 
        (1.267 × Feature4) - (1.615 × Feature50) - (1.408 × Feature53)
    """
    def __init__(self):
        self.intercept_ = -4.496
        self.features = ['Feature2', 'Feature48', 'Feature49', 'Feature4', 'Feature50', 'Feature53']
        self.coefficients = {
            'Feature2': 0.07,
            'Feature48': 0.676,
            'Feature49': 0.736,
            'Feature4': 1.267,
            'Feature50': -1.615,
            'Feature53': -1.408
        }
        
    def fit(self, X, y):
        # 模型已经预定义，不需要训练
        return self
        
    def predict_proba(self, X):
        # 确保X是DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.features)
            
        # 计算线性组合
        x = np.zeros(len(X))
        x += self.intercept_
        
        for feature, coef in self.coefficients.items():
            if feature in X.columns:
                x += coef * X[feature].values
            
        # 计算概率
        p_malignant = 1 / (1 + np.exp(-x))
        
        # 返回两列概率 [P(benign), P(malignant)]
        return np.column_stack((1 - p_malignant, p_malignant))
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

class MayoModel:
    """
    Mayo模型的实现
    P(malignant) = e^x / (1+e^x)
    x = -6.8272 + (0.0391 × Feature2) + (0.7917 × Feature3) + (1.3388 × Feature5) + 
        (0.1274 × Feature48) + (1.0407 × Feature49) + (0.7838 × Feature63)
    """
    def __init__(self):
        self.intercept_ = -6.8272
        self.features = ['Feature2', 'Feature3', 'Feature5', 'Feature48', 'Feature49', 'Feature63']
        self.coefficients = {
            'Feature2': 0.0391,
            'Feature3': 0.7917,
            'Feature5': 1.3388,
            'Feature48': 0.1274,
            'Feature49': 1.0407,
            'Feature63': 0.7838
        }
        
    def fit(self, X, y):
        # 模型已经预定义，不需要训练
        return self
        
    def predict_proba(self, X):
        # 确保X是DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.features)
            
        # 计算线性组合
        x = np.zeros(len(X))
        x += self.intercept_
        
        for feature, coef in self.coefficients.items():
            if feature in X.columns:
                x += coef * X[feature].values
            
        # 计算概率
        p_malignant = 1 / (1 + np.exp(-x))
        
        # 返回两列概率 [P(benign), P(malignant)]
        return np.column_stack((1 - p_malignant, p_malignant))
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

def evaluate_metrics(y_true, y_pred, y_pred_proba):
    """计算所有评估指标"""
    conf_matrix = confusion_matrix(y_true, y_pred)
    return {
        'acc': accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_pred_proba),
        'f1': f1_score(y_true, y_pred),
        'acc_0': conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]),
        'acc_1': conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
    }

def print_metrics(dataset_name, metrics):
    """打印评估指标"""
    print(f"{dataset_name}准确率 (Accuracy): {metrics['acc']:.4f}")
    print(f"{dataset_name} AUC: {metrics['auc']:.4f}")
    print(f"{dataset_name} F1分数: {metrics['f1']:.4f}")
    print(f"{dataset_name}类别0准确率: {metrics['acc_0']:.4f}")
    print(f"{dataset_name}类别1准确率: {metrics['acc_1']:.4f}")

def run_experiment_with_model(
    X,
    y,
    model_name,
    model_constructor,
    model_params={},
    base_path='./results'
):
    """
    使用指定模型运行10折交叉验证实验
    
    Parameters:
    -----------
    X : pd.DataFrame
        特征矩阵
    y : pd.Series
        目标变量
    model_name : str
        模型名称
    model_constructor : class
        模型构造函数
    model_params : dict
        模型参数
    base_path : str
        结果保存路径
    
    Returns:
    --------
    pd.DataFrame
        包含交叉验证分数的DataFrame
    """
    # 创建结果目录（如果不存在）
    os.makedirs(base_path, exist_ok=True)
    
    # 生成基于模型名称的实验名称
    exp_name = f"{model_name}-Experiment"
    
    print(f"\n=== {model_name} 模型 ===")
    print("数据形状:", X.shape)
    print("标签分布:\n", y.value_counts())
    
    # 转换数据为numpy数组
    X_values = X.copy()
    y_values = y.values.astype(np.int32)
    
    # ==============================
    # 交叉验证
    # ==============================
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_values), 1):
        X_train, X_test = X_values.iloc[train_idx], X_values.iloc[test_idx]
        y_train, y_test = y_values[train_idx], y_values[test_idx]
        
        print(f"\n折 {fold}")
        print("-" * 50)
        
        # 初始化并训练模型
        start_time = time.time()
        model = model_constructor(**model_params)
        model.fit(X_train, y_train)
        
        # 进行预测
        y_pred_proba = model.predict_proba(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        fold_time = time.time() - start_time
        
        # 计算指标
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        f1 = f1_score(y_test, y_pred)
        
        # 计算每类的准确率
        conf_matrix = confusion_matrix(y_test, y_pred)
        acc_0 = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
        acc_1 = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
        
        print(f"准确率: {acc:.4f}")
        print(f"ROC AUC: {auc:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"类别0准确率: {acc_0:.4f}")
        print(f"类别1准确率: {acc_1:.4f}")
        print(f"时间: {fold_time:.4f}秒")
        
        fold_scores.append({
            'fold': fold,
            'accuracy': acc,
            'auc': auc,
            'f1': f1,
            'acc_0': acc_0,
            'acc_1': acc_1,
            'time': fold_time
        })
    
    # ==============================
    # 汇总结果
    # ==============================
    scores_df = pd.DataFrame(fold_scores)
    
    # 保存结果
    scores_df.to_csv(f'{base_path}/{exp_name}.csv', index=False)
    
    # 计算并保存最终结果
    final_results = pd.DataFrame({
        'Metric': ['AUC', 'F1', 'ACC', 'ACC_0', 'ACC_1', 'Time'],
        'Mean': [
            scores_df['auc'].mean(),
            scores_df['f1'].mean(),
            scores_df['accuracy'].mean(),
            scores_df['acc_0'].mean(),
            scores_df['acc_1'].mean(),
            scores_df['time'].mean()
        ],
        'Std': [
            scores_df['auc'].std(),
            scores_df['f1'].std(),
            scores_df['accuracy'].std(),
            scores_df['acc_0'].std(),
            scores_df['acc_1'].std(),
            scores_df['time'].std()
        ]
    })
    final_results.to_csv(f'{base_path}/{exp_name}-Final.csv', index=False)
    
    # 打印结果
    print("\n最终结果:")
    print(f"平均测试AUC: {scores_df['auc'].mean():.4f} ± {scores_df['auc'].std():.4f}")
    print(f"平均测试F1: {scores_df['f1'].mean():.4f} ± {scores_df['f1'].std():.4f}")
    print(f"平均测试准确率: {scores_df['accuracy'].mean():.4f} ± {scores_df['accuracy'].std():.4f}")
    print(f"平均测试类别0准确率: {scores_df['acc_0'].mean():.4f} ± {scores_df['acc_0'].std():.4f}")
    print(f"平均测试类别1准确率: {scores_df['acc_1'].mean():.4f} ± {scores_df['acc_1'].std():.4f}")
    print(f"平均时间: {scores_df['time'].mean():.4f} ± {scores_df['time'].std():.4f}")
    
    # ==============================
    # 可视化结果
    # ==============================
    plt.figure(figsize=(15, 5))
    
    # 绘制指标
    plt.subplot(1, 3, 1)
    metrics = ['accuracy', 'auc', 'f1']
    for metric in metrics:
        plt.plot(scores_df['fold'], scores_df[metric], 'o-', label=metric.upper())
        plt.axhline(y=scores_df[metric].mean(), linestyle='--', alpha=0.3)
    plt.title('Performance Metrics Across Folds')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.legend()
    
    # 绘制每类准确率
    plt.subplot(1, 3, 2)
    plt.plot(scores_df['fold'], scores_df['acc_0'], 'bo-', label='Class 0 Accuracy')
    plt.plot(scores_df['fold'], scores_df['acc_1'], 'ro-', label='Class 1 Accuracy')
    plt.axhline(y=scores_df['acc_0'].mean(), color='b', linestyle='--', alpha=0.3)
    plt.axhline(y=scores_df['acc_1'].mean(), color='r', linestyle='--', alpha=0.3)
    plt.title('Per-class Accuracy Across Folds')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 绘制时间
    plt.subplot(1, 3, 3)
    plt.plot(scores_df['fold'], scores_df['time'], 'go-', label='Computation Time')
    plt.axhline(y=scores_df['time'].mean(), color='g', linestyle='--', alpha=0.3)
    plt.title('Computation Time Across Folds')
    plt.xlabel('Fold')
    plt.ylabel('Time (s)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{base_path}/{exp_name}-Plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return scores_df

class TabularModel(nn.Module):
    """
    用于表格数据分类的PyTorch模型，支持CORAL域适应
    """
    def __init__(self, input_dim, hidden_dims=[64, 32], output_dim=2, dropout=0.3):
        super(TabularModel, self).__init__()
        
        # 特征提取器（骨干网络）
        backbone_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            backbone_layers.append(nn.Linear(prev_dim, hidden_dim))
            backbone_layers.append(nn.BatchNorm1d(hidden_dim))
            backbone_layers.append(nn.ReLU())
            backbone_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
            
        self.backbone = nn.Sequential(*backbone_layers)
        
        # 分类器
        self.classifier = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, x, return_features=False):
        features = self.backbone(x)
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits

def compute_coral_loss(source_features, target_features):
    """
    计算CORAL损失，即源域和目标域特征的协方差矩阵之间的差异
    
    参数:
    - source_features: 源域特征张量 [batch_size, feature_dim]
    - target_features: 目标域特征张量 [batch_size, feature_dim]
    
    返回:
    - coral_loss: 源域和目标域之间的CORAL损失
    """
    device = source_features.device
    
    # 特征维度
    d = source_features.size(1)
    
    # 源域协方差
    source_features = source_features - torch.mean(source_features, dim=0, keepdim=True)
    source_cov = torch.matmul(source_features.t(), source_features) / (source_features.size(0) - 1)
    
    # 目标域协方差
    target_features = target_features - torch.mean(target_features, dim=0, keepdim=True)
    target_cov = torch.matmul(target_features.t(), target_features) / (target_features.size(0) - 1)
    
    # 计算Frobenius范数
    coral_loss = torch.norm(source_cov - target_cov, p='fro') ** 2
    coral_loss = coral_loss / (4 * d * d)
    
    return coral_loss

def train_with_coral(
    model,
    source_dataloader,
    target_dataloader,
    optimizer,
    device,
    epochs=100,
    lambda_coral=1.0,
    scheduler=None,
    early_stopping_patience=10,
    verbose=True
):
    """
    使用CORAL域适应训练模型
    
    参数:
    - model: 模型
    - source_dataloader: 源域数据加载器
    - target_dataloader: 目标域数据加载器
    - optimizer: 优化器
    - device: 设备
    - epochs: 训练轮数
    - lambda_coral: CORAL损失的权重
    - scheduler: 学习率调度器
    - early_stopping_patience: 早停耐心值
    - verbose: 是否打印训练过程
    
    返回:
    - model: 训练后的模型
    - history: 训练历史记录
    """
    # 将模型移至设备
    model = model.to(device)
    
    # 损失函数
    classification_criterion = nn.CrossEntropyLoss()
    
    # 训练历史
    history = {
        'train_loss': [],
        'classification_loss': [],
        'coral_loss': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # 早停
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # 训练模式
        model.train()
        
        # 重置指标
        train_loss = 0.0
        classification_loss_sum = 0.0
        coral_loss_sum = 0.0
        
        # 确保目标域数据加载器可以循环使用
        target_iter = iter(target_dataloader)
        
        # 训练一个周期
        for source_batch in source_dataloader:
            # 获取源域数据
            source_data, source_labels = source_batch
            source_data = source_data.to(device)
            source_labels = source_labels.to(device)
            
            # 获取目标域数据
            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_dataloader)
                target_batch = next(target_iter)
                
            target_data = target_batch[0].to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播（同时返回特征）
            source_logits, source_features = model(source_data, return_features=True)
            _, target_features = model(target_data, return_features=True)
            
            # 计算分类损失
            classification_loss = classification_criterion(source_logits, source_labels)
            
            # 计算CORAL损失
            coral_loss = compute_coral_loss(source_features, target_features)
            
            # 总损失
            loss = classification_loss + lambda_coral * coral_loss
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 更新指标
            train_loss += loss.item()
            classification_loss_sum += classification_loss.item()
            coral_loss_sum += coral_loss.item()
        
        # 计算平均损失
        avg_train_loss = train_loss / len(source_dataloader)
        avg_classification_loss = classification_loss_sum / len(source_dataloader)
        avg_coral_loss = coral_loss_sum / len(source_dataloader)
        
        # 更新学习率
        if scheduler:
            scheduler.step()
        
        # 更新历史
        history['train_loss'].append(avg_train_loss)
        history['classification_loss'].append(avg_classification_loss)
        history['coral_loss'].append(avg_coral_loss)
        
        # 验证
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in target_dataloader:
                data = data.to(device)
                labels = labels.to(device)
                
                outputs = model(data)
                loss = classification_criterion(outputs, labels)
                
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # 计算平均验证损失和准确率
        avg_val_loss = val_loss / len(target_dataloader)
        val_acc = correct / total
        
        # 更新历史
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        # 早停
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"早停: 在{epoch+1}轮后停止训练")
                break
        
        # 打印训练信息
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"轮次 {epoch+1}/{epochs}")
            print(f"训练损失: {avg_train_loss:.4f}")
            print(f"分类损失: {avg_classification_loss:.4f}")
            print(f"CORAL损失: {avg_coral_loss:.4f}")
            print(f"验证损失: {avg_val_loss:.4f}")
            print(f"验证准确率: {val_acc:.4f}")
    
    return model, history

def prepare_data_for_coral(X_source, y_source, X_target, y_target=None, test_size=0.2, batch_size=32):
    """
    准备用于CORAL域适应的数据
    
    参数:
    - X_source: 源域特征
    - y_source: 源域标签
    - X_target: 目标域特征
    - y_target: 目标域标签（可选，用于评估）
    - test_size: 测试集比例
    - batch_size: 批大小
    
    返回:
    - dataloaders: 包含数据加载器的字典
    - input_dim: 输入维度
    """
    # 标准化所有数据
    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_source)
    X_target_scaled = scaler.transform(X_target)
    
    # 将源域数据分为训练集和验证集
    X_source_train, X_source_val, y_source_train, y_source_val = train_test_split(
        X_source_scaled, y_source, test_size=test_size, random_state=42
    )
    
    # 创建数据集
    source_train_dataset = TensorDataset(
        torch.FloatTensor(X_source_train),
        torch.LongTensor(y_source_train)
    )
    
    source_val_dataset = TensorDataset(
        torch.FloatTensor(X_source_val),
        torch.LongTensor(y_source_val)
    )
    
    # 默认情况下，我们假设目标域没有标签，但如果提供了标签，则用于评估
    if y_target is not None:
        target_dataset = TensorDataset(
            torch.FloatTensor(X_target_scaled),
            torch.LongTensor(y_target)
        )
    else:
        # 创建无标签的目标域数据集（使用占位符标签）
        dummy_labels = np.zeros(len(X_target_scaled))
        target_dataset = TensorDataset(
            torch.FloatTensor(X_target_scaled),
            torch.LongTensor(dummy_labels)
        )
    
    # 创建数据加载器
    source_train_loader = DataLoader(
        source_train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    source_val_loader = DataLoader(
        source_val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    dataloaders = {
        'source_train': source_train_loader,
        'source_val': source_val_loader,
        'target': target_loader
    }
    
    # 输入维度
    input_dim = X_source.shape[1]
    
    return dataloaders, input_dim

def evaluate_coral_model(model, dataloader, device):
    """
    评估模型在给定数据加载器上的性能
    
    参数:
    - model: 模型
    - dataloader: 数据加载器
    - device: 设备
    
    返回:
    - metrics: 包含评估指标的字典
    """
    model.eval()
    
    # 初始化指标
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    probabilities = []
    
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            labels = labels.to(device)
            
            outputs = model(data)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 收集预测和真实标签
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    # 计算指标
    accuracy = correct / total
    
    # 转换为NumPy数组以方便计算
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    probabilities = np.array(probabilities)
    
    # 计算每类的准确率
    class_0_mask = true_labels == 0
    class_1_mask = true_labels == 1
    
    acc_0 = np.sum(predictions[class_0_mask] == true_labels[class_0_mask]) / np.sum(class_0_mask)
    acc_1 = np.sum(predictions[class_1_mask] == true_labels[class_1_mask]) / np.sum(class_1_mask)
    
    # 计算AUC（如果有多于一个类）
    if len(np.unique(true_labels)) > 1:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(true_labels, probabilities[:, 1])
    else:
        auc = float('nan')
    
    # 计算F1分数
    from sklearn.metrics import f1_score
    f1 = f1_score(true_labels, predictions)
    
    return {
        'acc': accuracy,
        'auc': auc,
        'f1': f1,
        'acc_0': acc_0,
        'acc_1': acc_1
    }

def run_coral_adaptation_experiment(
    X_source,
    y_source,
    X_target,
    y_target,
    model_name='TabPFN-CORAL',
    lambda_coral=1.0,
    align_epochs=100,
    tabpfn_params={'device': 'cuda', 'max_time': 60, 'random_state': 42},
    base_path='./results_coral'
):
    """
    运行带有CORAL域适应的TabPFN实验
    
    参数:
    - X_source: 源域特征
    - y_source: 源域标签
    - X_target: 目标域特征
    - y_target: 目标域标签
    - model_name: 模型名称
    - lambda_coral: CORAL损失权重
    - align_epochs: AlignNet训练轮数
    - tabpfn_params: TabPFN参数
    - base_path: 结果保存路径
    
    返回:
    - 评估指标
    """
    logging.info(f"\n=== {model_name} Model (lambda_coral={lambda_coral}) ===")
    
    # 创建结果目录
    os.makedirs(base_path, exist_ok=True)
    
    # 数据标准化
    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_source)
    X_target_scaled = scaler.transform(X_target)
    
    # 分析源域和目标域的特征分布差异
    logging.info("Analyzing domain differences before alignment...")
    source_mean = np.mean(X_source_scaled, axis=0)
    target_mean = np.mean(X_target_scaled, axis=0)
    source_std = np.std(X_source_scaled, axis=0)
    target_std = np.std(X_target_scaled, axis=0)
    
    mean_diff = np.mean(np.abs(source_mean - target_mean))
    std_diff = np.mean(np.abs(source_std - target_std))
    logging.info(f"Initial domain difference: Mean diff={mean_diff:.6f}, Std diff={std_diff:.6f}")
    
    # 初始化TabPFN模型（固定，不训练）
    logging.info("Initializing TabPFN model...")
    tabpfn_model = AutoTabPFNClassifier(**tabpfn_params)
    
    # 在源域数据上训练TabPFN
    logging.info("Training TabPFN on source domain data...")
    start_time = time.time()
    tabpfn_model.fit(X_source_scaled, y_source)
    tabpfn_time = time.time() - start_time
    logging.info(f"TabPFN training completed in {tabpfn_time:.2f} seconds")
    
    # 在目标域上进行直接评估（未对齐）
    logging.info("\nEvaluating TabPFN directly on target domain (without alignment)...")
    y_target_pred_direct = tabpfn_model.predict(X_target_scaled)
    y_target_proba_direct = tabpfn_model.predict_proba(X_target_scaled)
    
    # 计算原始TabPFN在目标域上的性能
    direct_metrics = {
        'acc': accuracy_score(y_target, y_target_pred_direct),
        'auc': roc_auc_score(y_target, y_target_proba_direct[:, 1]),
        'f1': f1_score(y_target, y_target_pred_direct)
    }
    
    # 计算每类准确率
    conf_matrix = confusion_matrix(y_target, y_target_pred_direct)
    direct_metrics['acc_0'] = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) if (conf_matrix[0, 0] + conf_matrix[0, 1]) > 0 else 0
    direct_metrics['acc_1'] = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1]) if (conf_matrix[1, 0] + conf_matrix[1, 1]) > 0 else 0
    
    logging.info(f"Direct prediction - Accuracy: {direct_metrics['acc']:.4f}, AUC: {direct_metrics['auc']:.4f}, F1: {direct_metrics['f1']:.4f}")
    logging.info(f"Direct prediction - Class 0 Acc: {direct_metrics['acc_0']:.4f}, Class 1 Acc: {direct_metrics['acc_1']:.4f}")
    
    # 初始化并训练AlignNet
    logging.info("\nInitializing AlignNet model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    align_model = AlignNet(input_dim=X_source.shape[1])
    
    logging.info(f"Training AlignNet on source and target domain data (lambda_coral={lambda_coral})...")
    start_time = time.time()
    align_model, align_history = train_alignnet(
        source_data=X_source_scaled,
        target_data=X_target_scaled,
        model=align_model,
        device=device,
        epochs=align_epochs,
        lambda_coral=lambda_coral,
        verbose=True
    )
    align_time = time.time() - start_time
    logging.info(f"AlignNet training completed in {align_time:.2f} seconds")
    
    # 获取最终混合参数
    with torch.no_grad():
        final_alpha = torch.sigmoid(align_model.mix_param).item()
    logging.info(f"Final alpha (mix parameter): {final_alpha:.4f}")
    
    # 在目标域上进行评估
    logging.info("\nEvaluating model on target domain (with alignment)...")
    
    # 源域预测（直接使用TabPFN）
    start_time = time.time()
    y_source_pred, y_source_proba = predict_with_alignment(
        tabpfn_model, align_model, X_source_scaled, X_source_scaled, is_source_domain=True
    )
    source_metrics = {
        'acc': accuracy_score(y_source, y_source_pred),
        'auc': roc_auc_score(y_source, y_source_proba[:, 1]),
        'f1': f1_score(y_source, y_source_pred)
    }
    
    # 目标域预测（使用AlignNet对齐）
    y_target_pred, y_target_proba = predict_with_alignment(
        tabpfn_model, align_model, X_source_scaled, X_target_scaled, is_source_domain=False
    )
    inference_time = time.time() - start_time
    
    # 计算目标域指标
    target_metrics = {
        'acc': accuracy_score(y_target, y_target_pred),
        'auc': roc_auc_score(y_target, y_target_proba[:, 1]),
        'f1': f1_score(y_target, y_target_pred)
    }
    
    # 计算混淆矩阵和每类准确率
    conf_matrix = confusion_matrix(y_target, y_target_pred)
    acc_0 = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) if (conf_matrix[0, 0] + conf_matrix[0, 1]) > 0 else 0
    acc_1 = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1]) if (conf_matrix[1, 0] + conf_matrix[1, 1]) > 0 else 0
    
    target_metrics['acc_0'] = acc_0
    target_metrics['acc_1'] = acc_1
    
    # 打印结果
    logging.info("\nSource Domain Evaluation Results:")
    logging.info(f"Accuracy: {source_metrics['acc']:.4f}")
    logging.info(f"AUC: {source_metrics['auc']:.4f}")
    logging.info(f"F1: {source_metrics['f1']:.4f}")
    
    logging.info("\nTarget Domain Evaluation Results:")
    logging.info(f"Accuracy: {target_metrics['acc']:.4f}")
    logging.info(f"AUC: {target_metrics['auc']:.4f}")
    logging.info(f"F1: {target_metrics['f1']:.4f}")
    logging.info(f"Class 0 Accuracy: {target_metrics['acc_0']:.4f}")
    logging.info(f"Class 1 Accuracy: {target_metrics['acc_1']:.4f}")
    logging.info(f"Inference Time: {inference_time:.4f} seconds")
    
    # 比较对齐前后的性能
    logging.info("\nPerformance Improvement with CORAL Alignment:")
    logging.info(f"Accuracy: {direct_metrics['acc']:.4f} -> {target_metrics['acc']:.4f} ({target_metrics['acc']-direct_metrics['acc']:.4f})")
    logging.info(f"AUC: {direct_metrics['auc']:.4f} -> {target_metrics['auc']:.4f} ({target_metrics['auc']-direct_metrics['auc']:.4f})")
    logging.info(f"F1: {direct_metrics['f1']:.4f} -> {target_metrics['f1']:.4f} ({target_metrics['f1']-direct_metrics['f1']:.4f})")
    
    # 保存模型
    model_path = f"{base_path}/{model_name}_lambda_{lambda_coral}.pt"
    torch.save(align_model.state_dict(), model_path)
    logging.info(f"AlignNet model saved to: {model_path}")
    
    # 绘制CORAL损失曲线
    plt.figure(figsize=(15, 5))
    
    # 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(align_history['coral_loss'], label='CORAL Loss')
    plt.title(f'CORAL Loss (lambda={lambda_coral})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 混合参数曲线
    plt.subplot(1, 3, 2)
    plt.plot(align_history['alpha_values'], label='Alpha')
    plt.title('Feature Mixing Parameter')
    plt.xlabel('Epochs')
    plt.ylabel('Alpha')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 类别分布对比
    plt.subplot(1, 3, 3)
    labels = ['Direct', 'With CORAL']
    class0_accs = [direct_metrics['acc_0'], target_metrics['acc_0']]
    class1_accs = [direct_metrics['acc_1'], target_metrics['acc_1']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, class0_accs, width, label='Class 0')
    plt.bar(x + width/2, class1_accs, width, label='Class 1')
    
    plt.xlabel('Method')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy Comparison')
    plt.xticks(x, labels)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{base_path}/{model_name}_lambda_{lambda_coral}_analysis.png", dpi=300)
    plt.close()
    
    # 返回完整指标
    return {
        'source': source_metrics,
        'target': target_metrics,
        'direct': direct_metrics,
        'alpha': final_alpha,
        'times': {
            'tabpfn': tabpfn_time,
            'align': align_time,
            'inference': inference_time
        }
    }

# 修改AlignNet类，改进特征对齐方法
class AlignNet(nn.Module):
    """
    用于特征对齐的MLP网络，目的是将目标域的特征分布与源域对齐
    改进版本：保留部分原始特征信息，避免过度对齐
    """
    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        super(AlignNet, self).__init__()
        # 保留原始输入的恒等映射
        self.identity = nn.Identity()
        
        # 特征转换网络
        self.align_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # 初始混合参数为0.5（可学习参数）
        self.mix_param = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        # 获取原始特征
        identity_feat = self.identity(x)
        # 获取对齐后的特征
        aligned_feat = self.align_network(x)
        
        # 使用sigmoid确保混合参数在0到1之间
        alpha = torch.sigmoid(self.mix_param)
        # 混合原始特征和对齐特征
        return alpha * aligned_feat + (1 - alpha) * identity_feat

# 修改train_alignnet函数，确保正确使用lambda_coral参数
def train_alignnet(
    source_data, 
    target_data, 
    model,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    batch_size=32,
    epochs=100,
    lr=0.001,
    lambda_coral=1.0,
    early_stopping_patience=10,
    verbose=True
):
    """
    训练AlignNet对齐模型
    
    参数:
    - source_data: 源域数据
    - target_data: 目标域数据
    - model: AlignNet模型
    - device: 计算设备
    - batch_size: 批大小
    - epochs: 训练轮数
    - lr: 学习率
    - lambda_coral: CORAL损失权重
    - early_stopping_patience: 早停耐心值
    - verbose: 是否打印训练进度
    
    返回:
    - model: 训练好的模型
    - history: 训练历史
    """
    # 准备数据
    source_tensor = torch.FloatTensor(source_data).to(device)
    target_tensor = torch.FloatTensor(target_data).to(device)
    
    # 移动模型到设备
    model = model.to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 记录历史
    history = {
        'loss': [],
        'coral_loss': [],
        'best_loss': float('inf'),
        'alpha_values': []  # 记录混合参数
    }
    
    # 早停计数器
    patience_counter = 0
    best_model_state = None
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_coral = 0.0
        
        # 计算源域和目标域的批次数
        n_batches = max(1, min(len(source_tensor), len(target_tensor)) // batch_size)
        
        for i in range(n_batches):
            # 获取批次数据
            s_start, s_end = i * batch_size, min((i + 1) * batch_size, len(source_tensor))
            t_start, t_end = i * batch_size, min((i + 1) * batch_size, len(target_tensor))
            
            # 如果已经用完了所有数据，就重新开始
            if s_start >= len(source_tensor):
                s_start, s_end = 0, min(batch_size, len(source_tensor))
            if t_start >= len(target_tensor):
                t_start, t_end = 0, min(batch_size, len(target_tensor))
                
            source_batch = source_tensor[s_start:s_end]
            target_batch = target_tensor[t_start:t_end]
            
            # 前向传播
            target_aligned = model(target_batch)
            
            # 计算CORAL损失，正确应用lambda_coral权重
            coral_loss_value = compute_coral_loss(source_batch, target_aligned)
            loss = lambda_coral * coral_loss_value
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录损失
            epoch_loss += loss.item()
            epoch_coral += coral_loss_value.item()
        
        # 计算平均损失
        avg_loss = epoch_loss / n_batches
        avg_coral = epoch_coral / n_batches
        
        # 获取当前混合参数
        with torch.no_grad():
            current_alpha = torch.sigmoid(model.mix_param).item()
            history['alpha_values'].append(current_alpha)
        
        # 记录历史
        history['loss'].append(avg_loss)
        history['coral_loss'].append(avg_coral)
        
        # 早停检查
        if avg_loss < history['best_loss']:
            history['best_loss'] = avg_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            if verbose:
                logging.info(f"Early stopping triggered at epoch {epoch+1}")
            break
        
        # 打印训练进度
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, CORAL Loss: {avg_coral:.6f}, Alpha: {current_alpha:.4f}")
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

# 修改predict_with_alignment函数，增加验证步骤
def predict_with_alignment(
    tabpfn_model,
    align_model,
    X_source,
    X_target,
    is_source_domain=False,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    使用AlignNet对齐后的特征进行TabPFN预测
    
    参数:
    - tabpfn_model: TabPFN模型
    - align_model: AlignNet模型
    - X_source: 源域数据（用于校准）
    - X_target: 目标域数据
    - is_source_domain: 是否为源域数据（如果是，则跳过对齐）
    
    返回:
    - y_pred: 预测标签
    - y_proba: 预测概率
    """
    # 如果是源域数据，直接使用TabPFN预测
    if is_source_domain:
        return tabpfn_model.predict(X_target), tabpfn_model.predict_proba(X_target)
    
    # 否则，首先通过AlignNet对目标域数据进行对齐
    align_model.eval()
    with torch.no_grad():
        X_target_tensor = torch.FloatTensor(X_target).to(device)
        X_target_aligned = align_model(X_target_tensor).cpu().numpy()
        
        # 检查对齐前后的特征差异
        mean_diff = np.mean(np.abs(X_target - X_target_aligned))
        std_diff = np.std(np.abs(X_target - X_target_aligned))
        
        # 记录对齐前后的特征差异
        logging.info(f"Feature alignment: Mean diff={mean_diff:.6f}, Std diff={std_diff:.6f}")
        
        # 获取当前混合参数
        current_alpha = torch.sigmoid(align_model.mix_param).item()
        logging.info(f"Current alpha (mix parameter): {current_alpha:.4f}")
        
        # 如果差异太大，可能会导致过度对齐
        if mean_diff > 1.0:
            logging.warning(f"Warning: Large feature transformation detected (mean_diff={mean_diff:.6f})")
    
    # 使用对齐后的特征进行TabPFN预测
    y_pred = tabpfn_model.predict(X_target_aligned)
    y_proba = tabpfn_model.predict_proba(X_target_aligned)
    
    # 检查预测分布是否均衡
    unique_labels, counts = np.unique(y_pred, return_counts=True)
    logging.info(f"Prediction distribution: {dict(zip(unique_labels, counts))}")
    
    return y_pred, y_proba

# ==============================
# 主程序
# ==============================
if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.manual_seed(42)
    
    # 指定设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    
    # 加载所有数据集
    logging.info("\nLoading datasets...")
    logging.info("1. Loading AI4healthcare.xlsx (A)...")
    df_ai4health = pd.read_excel("data/AI4healthcare.xlsx")
    
    logging.info("2. Loading HenanCancerHospital_features63_58.xlsx (B)...")
    df_henan = pd.read_excel("data/HenanCancerHospital_features63_58.xlsx")
    
    logging.info("3. Loading GuangzhouMedicalHospital_features23_no_nan.xlsx (C)...")
    df_guangzhou = pd.read_excel("data/GuangzhouMedicalHospital_features23_no_nan.xlsx")

    # 使用指定的23个特征
    selected_features = [
        'Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5',
        'Feature14', 'Feature15', 'Feature17', 'Feature22',
        'Feature39', 'Feature42', 'Feature43', 'Feature45',
        'Feature46', 'Feature47', 'Feature48', 'Feature49',
        'Feature50', 'Feature52', 'Feature53', 'Feature56',
        'Feature57', 'Feature63'
    ]

    logging.info("\n=== Feature Information ===")
    logging.info(f"Number of selected features: {len(selected_features)}")
    logging.info(f"Selected features list: {selected_features}")

    # 检查每个数据集中是否有所有选定的特征
    for dataset_name, dataset in [
        ("AI4health", df_ai4health), 
        ("Henan", df_henan), 
        ("Guangzhou", df_guangzhou)
    ]:
        missing_features = [f for f in selected_features if f not in dataset.columns]
        if missing_features:
            logging.warning(f"Warning: {dataset_name} missing the following features: {missing_features}")
        else:
            logging.info(f"{dataset_name} contains all selected features")

    # 使用共同特征准备数据
    X_ai4health = df_ai4health[selected_features].copy()
    y_ai4health = df_ai4health["Label"].copy()
    
    X_henan = df_henan[selected_features].copy()
    y_henan = df_henan["Label"].copy()
    
    X_guangzhou = df_guangzhou[selected_features].copy()
    y_guangzhou = df_guangzhou["Label"].copy()

    # 创建结果目录
    os.makedirs('./results_tabpfn_coral', exist_ok=True)

    # 运行CORAL域适应实验
    logging.info("\n\n=== Running TabPFN-CORAL Domain Adaptation Experiments ===")
    
    # 定义域适应配置 (从A到B/C)
    coral_configs = [
        {
            'name': 'A_to_B',
            'source_name': 'A_AI4health',
            'target_name': 'B_Henan',
            'X_source': X_ai4health,
            'y_source': y_ai4health,
            'X_target': X_henan,
            'y_target': y_henan
        },
        {
            'name': 'A_to_C',
            'source_name': 'A_AI4health',
            'target_name': 'C_Guangzhou',
            'X_source': X_ai4health,
            'y_source': y_ai4health,
            'X_target': X_guangzhou,
            'y_target': y_guangzhou
        }
    ]
    
    # 存储所有实验结果
    all_results = []
    
    # 运行不同的CORAL权重实验
    for lambda_coral in [0.1, 1.0, 5.0]:
        for config in coral_configs:
            logging.info(f"\n\n{'='*50}")
            logging.info(f"Domain Adaptation: {config['source_name']} → {config['target_name']}, lambda_coral={lambda_coral}")
            logging.info(f"{'='*50}")
            
            # 运行CORAL域适应实验
            metrics = run_coral_adaptation_experiment(
                X_source=config['X_source'],
                y_source=config['y_source'],
                X_target=config['X_target'],
                y_target=config['y_target'],
                model_name=f"TabPFN-CORAL_{config['name']}",
                lambda_coral=lambda_coral,
                align_epochs=50,  # 减少轮数以加快实验
                base_path='./results_tabpfn_coral'
            )
            
            # 保存结果
            result = {
                'source': config['source_name'],
                'target': config['target_name'],
                'lambda_coral': lambda_coral,
                'source_acc': metrics['source']['acc'],
                'source_auc': metrics['source']['auc'],
                'source_f1': metrics['source']['f1'],
                'target_acc': metrics['target']['acc'],
                'target_auc': metrics['target']['auc'],
                'target_f1': metrics['target']['f1'],
                'target_acc_0': metrics['target']['acc_0'],
                'target_acc_1': metrics['target']['acc_1'],
                'tabpfn_time': metrics['times']['tabpfn'],
                'align_time': metrics['times']['align'],
                'inference_time': metrics['times']['inference']
            }
            all_results.append(result)
    
    # 创建结果表格
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('./results_tabpfn_coral/all_results.csv', index=False)
    logging.info("Results saved to ./results_tabpfn_coral/all_results.csv")
    
    # 可视化不同lambda_coral的效果
    logging.info("Generating visualization for different lambda_coral values...")
    plt.figure(figsize=(15, 10))
    
    # 为每个源域-目标域组合绘制图表
    for i, (source, target) in enumerate([('A_AI4health', 'B_Henan'), ('A_AI4health', 'C_Guangzhou')]):
        plt.subplot(2, 2, i*2+1)
        data = results_df[(results_df['source'] == source) & (results_df['target'] == target)]
        
        plt.plot(data['lambda_coral'], data['target_acc'], 'o-', label='Accuracy')
        plt.plot(data['lambda_coral'], data['target_auc'], 's-', label='AUC')
        plt.plot(data['lambda_coral'], data['target_f1'], '^-', label='F1')
        
        plt.title(f'{source} → {target}')
        plt.xlabel('CORAL Loss Weight (λ)')
        plt.ylabel('Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 每类准确率
        plt.subplot(2, 2, i*2+2)
        plt.plot(data['lambda_coral'], data['target_acc_0'], 'o-', label='Class 0 Accuracy')
        plt.plot(data['lambda_coral'], data['target_acc_1'], 's-', label='Class 1 Accuracy')
        
        plt.title(f'{source} → {target} (Per-class Accuracy)')
        plt.xlabel('CORAL Loss Weight (λ)')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./results_tabpfn_coral/lambda_coral_comparison.png', dpi=300)
    plt.close()
    logging.info("Visualization saved to ./results_tabpfn_coral/lambda_coral_comparison.png")
    
    # 与原始TabPFN比较
    logging.info("\n\n=== Comparing TabPFN-CORAL with Original TabPFN ===")
    
    # 在原始TabPFN上运行实验
    tabpfn_results = []
    
    for config in coral_configs:
        logging.info(f"\n{'='*50}")
        logging.info(f"Original TabPFN: Training on {config['source_name']}, Testing on {config['target_name']}")
        logging.info(f"{'='*50}")
        
        # 初始化并训练TabPFN
        tabpfn_model = AutoTabPFNClassifier(device='cuda', max_time=60, random_state=42)
        
        # 数据标准化
        scaler = StandardScaler()
        X_source_scaled = scaler.fit_transform(config['X_source'])
        X_target_scaled = scaler.transform(config['X_target'])
        
        # 训练和评估
        start_time = time.time()
        tabpfn_model.fit(X_source_scaled, config['y_source'])
        train_time = time.time() - start_time
        
        # 预测
        start_time = time.time()
        y_pred = tabpfn_model.predict(X_target_scaled)
        y_proba = tabpfn_model.predict_proba(X_target_scaled)
        inference_time = time.time() - start_time
        
        # 评估
        metrics = evaluate_metrics(config['y_target'], y_pred, y_proba[:, 1])
        
        logging.info(f"Accuracy: {metrics['acc']:.4f}")
        logging.info(f"AUC: {metrics['auc']:.4f}")
        logging.info(f"F1: {metrics['f1']:.4f}")
        logging.info(f"Class 0 Accuracy: {metrics['acc_0']:.4f}")
        logging.info(f"Class 1 Accuracy: {metrics['acc_1']:.4f}")
        
        # 保存结果
        tabpfn_results.append({
            'source': config['source_name'],
            'target': config['target_name'],
            'method': 'Original TabPFN',
            'acc': metrics['acc'],
            'auc': metrics['auc'],
            'f1': metrics['f1'],
            'acc_0': metrics['acc_0'],
            'acc_1': metrics['acc_1'],
            'train_time': train_time,
            'inference_time': inference_time
        })
    
    # 加入CORAL结果用于比较
    for _, row in results_df.iterrows():
        tabpfn_results.append({
            'source': row['source'],
            'target': row['target'],
            'method': f"TabPFN-CORAL (λ={row['lambda_coral']})",
            'acc': row['target_acc'],
            'auc': row['target_auc'],
            'f1': row['target_f1'],
            'acc_0': row['target_acc_0'],
            'acc_1': row['target_acc_1'],
            'train_time': row['tabpfn_time'] + row['align_time'],
            'inference_time': row['inference_time']
        })
    
    # 创建比较表格
    comparison_df = pd.DataFrame(tabpfn_results)
    comparison_df.to_csv('./results_tabpfn_coral/method_comparison.csv', index=False)
    logging.info("Comparison results saved to ./results_tabpfn_coral/method_comparison.csv")
    
    # 可视化方法比较
    logging.info("Generating visualization for method comparison...")
    for target in ['B_Henan', 'C_Guangzhou']:
        plt.figure(figsize=(12, 6))
        
        # 筛选特定目标域的数据
        target_data = comparison_df[comparison_df['target'] == target]
        
        # 绘制不同指标的对比
        metrics = ['acc', 'auc', 'f1']
        labels = ['Accuracy', 'AUC', 'F1 Score']
        
        for i, (metric, label) in enumerate(zip(metrics, labels)):
            plt.subplot(1, 3, i+1)
            
            methods = target_data['method'].values
            values = target_data[metric].values
            
            # 绘制条形图
            bars = plt.bar(range(len(methods)), values)
            
            # 添加值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
            
            plt.title(f'{label} Comparison')
            plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
            plt.ylim(0, max(values) * 1.15)
            plt.grid(axis='y', alpha=0.3)
        
        plt.suptitle(f'Method Comparison (Target Domain: {target})', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.savefig(f'./results_tabpfn_coral/method_comparison_{target}.png', dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Method comparison for {target} saved to ./results_tabpfn_coral/method_comparison_{target}.png")
    
    logging.info("\nAll results saved to ./results_tabpfn_coral/ directory")