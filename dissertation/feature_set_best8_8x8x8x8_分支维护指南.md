# feature_set_best8_8x8x8x8 分支维护指南

## 📋 分支概述

### 分支信息
- **分支名称**: `feature_set_best8_8x8x8x8`
- **基于commit**: `2cc7ab2d9e44533cdf23c505bc157b06467df660` (2025-09-04)
- **参数格式**: 使用 `--feature_set` (旧版格式)
- **特征集**: BEST_8_FEATURES = ['Feature63', 'Feature2', 'Feature46', 'Feature61', 'Feature56', 'Feature42', 'Feature39', 'Feature43']
- **用途**: 复现原始8:8:8:8配置的结果，与main分支的`feature_type`格式区分

### 性能基准
- **PANDA_NoUDA AUC**: ~0.698
- **TCA AUC**: ~0.71
- **期望结果**: 应与 `complete_analysis_20250904_222331_8:8:8:8` 目录中的结果一致

---

## 🖥️ 服务器端操作

### 1. 检查当前状态
```bash
# 检查当前分支
git branch
git log --oneline -1  # 应该显示 2cc7ab2

# 验证特征集配置
cd uda_medical_imbalance_project
python -c "from config.settings import BEST_8_FEATURES; print('BEST_8_FEATURES:', BEST_8_FEATURES)"
```

### 2. 运行分析
```bash
# 方式1: 直接运行
cd uda_medical_imbalance_project
python scripts/run_complete_analysis.py --feature_set best8

# 方式2: 使用job_script.sh (需要修改第35行)
cd ..
vim job_script.sh  # 第35行改为: python scripts/run_complete_analysis.py --feature_set best8
sbatch job_script.sh
```

### 3. 修改代码和配置
```bash
# 确保在正确分支
git checkout feature_set_best8_8x8x8x8

# 修改文件 (示例: 修复Paper LR问题)
vim uda_medical_imbalance_project/scripts/run_complete_analysis.py
# 或
vim uda_medical_imbalance_project/config/settings.py

# 添加修改
git add .
git commit -m "Fix: [具体问题描述] for feature_set branch"
```

### 4. 推送更新
```bash
# 推送本地修改到远程
git push origin feature_set_best8_8x8x8x8

# 如果是首次推送该分支
git push -u origin feature_set_best8_8x8x8x8
```

### 5. 紧急修复流程
```bash
# 如果配置被意外重置
git show 2cc7ab2:uda_medical_imbalance_project/config/settings.py > uda_medical_imbalance_project/config/settings.py

# 如果脚本被意外修改
git checkout 2cc7ab2 -- uda_medical_imbalance_project/scripts/run_complete_analysis.py
```

---

## 💻 客户端操作

### 1. 同步远程更新
```bash
# 1. 切换到主分支并拉取最新代码
git checkout main
git fetch origin
git pull origin main

# 2. 切换到feature_set分支
git checkout feature_set_best8_8x8x8x8

# 3. 拉取该分支的最新更新
git pull origin feature_set_best8_8x8x8x8
```

### 2. 本地修改和测试
```bash
# 在feature_set分支上进行修改
git checkout feature_set_best8_8x8x8x8

# 修改配置或代码
vim uda_medical_imbalance_project/config/settings.py
vim uda_medical_imbalance_project/scripts/run_complete_analysis.py

# 本地测试 (需要确保数据路径正确)
cd uda_medical_imbalance_project
python scripts/run_complete_analysis.py --feature_set best8
```

### 3. 推送到远程供服务器使用
```bash
# 提交本地修改
git add .
git commit -m "Local fix: [具体修改描述]"

# 推送到远程
git push origin feature_set_best8_8x8x8x8
```

### 4. 切换回main分支
```bash
git checkout main
```

---

## 🔧 常见问题和解决方案

### 问题1: 配置文件被重置
**症状**: BEST_8_FEATURES包含Feature5而不是Feature39

**解决方案**:
```bash
# 服务器上
git checkout feature_set_best8_8x8x8x8
git show 2cc7ab2:uda_medical_imbalance_project/config/settings.py > uda_medical_imbalance_project/config/settings.py
```

### 问题2: 参数格式错误
**症状**: 脚本不识别 `--feature_set` 参数

**解决方案**:
```bash
# 确保在正确的分支，并使用旧版脚本
git checkout 2cc7ab2 -- uda_medical_imbalance_project/scripts/run_complete_analysis.py
```

### 问题3: Paper LR或其他基线方法问题
**解决方案**:
```bash
# 检查相关代码
grep -r "Paper_LR\|paper_lr" uda_medical_imbalance_project/

# 修复代码并提交
vim [相关文件]
git add .
git commit -m "Fix Paper LR implementation for feature_set branch"
git push origin feature_set_best8_8x8x8x8
```

### 问题4: 结果不一致
**症状**: AUC结果显著低于0.693

**检查清单**:
- [ ] 确认在 `feature_set_best8_8x8x8x8` 分支
- [ ] 确认使用 `--feature_set best8` 参数
- [ ] 确认BEST_8_FEATURES包含Feature39而非Feature5
- [ ] 确认随机种子为42
- [ ] 确认数据文件路径正确

---

## 📊 验证清单

每次修改后，请验证以下内容：

### 配置验证
```bash
git log --oneline -1  # 应该基于2cc7ab2
cd uda_medical_imbalance_project
python -c "from config.settings import BEST_8_FEATURES; print('Length:', len(BEST_8_FEATURES))"
```

### 参数验证
```bash
python scripts/run_complete_analysis.py --help  # 应该有 --feature_set 选项
```

### 结果验证
- PANDA_NoUDA AUC 应在 0.69-0.70 范围内
- TCA AUC 应在 0.70-0.72 范围内
- 结果应与 `complete_analysis_20250904_222331_8:8:8:8` 一致

---

## 🔄 常用命令速查

### 服务器端
```bash
# 切换分支
git checkout feature_set_best8_8x8x8x8

# 还原配置
git show 2cc7ab2:uda_medical_imbalance_project/config/settings.py > uda_medical_imbalance_project/config/settings.py

# 运行分析
cd uda_medical_imbalance_project
python scripts/run_complete_analysis.py --feature_set best8

# 提交推送
git add . && git commit -m "Fix description" && git push origin feature_set_best8_8x8x8x8
```

### 客户端
```bash
# 同步更新
git checkout feature_set_best8_8x8x8x8 && git pull origin feature_set_best8_8x8x8x8

# 切换回main
git checkout main
```

---

## 📞 联系和备注

- **维护人员**: 开发团队
- **最后更新**: 2025-11-17
- **重要提醒**: 该分支独立于main分支，请勿合并到main分支
- **数据目录**: 注意根据服务器环境调整数据文件路径
- **依赖库**: 确保ADAPT等依赖库已正确安装

---

*本文档用于维护feature_set_best8_8x8x8x8分支，确保8:8:8:8配置的正确复现。*