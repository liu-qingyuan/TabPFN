# requirements.txt 文档

## 概述

项目依赖管理文件，列出了运行项目所需的所有Python包。

## 核心依赖

### 机器学习框架
- `numpy`: 数值计算基础库
- `pandas`: 数据处理和分析
- `scikit-learn`: 机器学习算法库
- `torch`: PyTorch深度学习框架

### 可视化
- `matplotlib`: 基础绘图库
- `seaborn`: 统计可视化库

### TabPFN相关
- `tabpfn`: 原生TabPFN模型
- `tabpfn-extensions`: AutoTabPFN扩展（可选）

### 其他工具
- `openpyxl`: Excel文件读写
- `scipy`: 科学计算库

## 安装方法

### 基础安装
```bash
pip install -r requirements.txt
```

### 开发环境安装
```bash
pip install -r requirements.txt
pip install pytest  # 用于测试
```

### GPU支持
```bash
# 如果需要GPU支持，安装CUDA版本的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 版本兼容性

- Python >= 3.8
- 建议使用虚拟环境
- 某些包可能需要特定版本，详见requirements.txt

## 可选依赖

某些高级功能可能需要额外的包：
- `optuna`: 超参数优化（TunedTabPFN）
- `plotly`: 交互式可视化
- `jupyter`: Jupyter notebook支持 