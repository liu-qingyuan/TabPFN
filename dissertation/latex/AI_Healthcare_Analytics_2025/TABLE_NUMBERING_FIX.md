# 表格编号修复说明 (Table Numbering Fix)

## 问题描述
在将LaTeX转换为DOCX时，表格目录中"Table 1", "Table 2"等编号没有正确显示。

## 问题原因分析
1. **缺少表格编号支持插件**：原始命令缺少表格自动编号功能
2. **LaTeX格式不支持native_numbering扩展**：LaTeX输入不支持此特定扩展
3. **版本兼容性问题**：某些pandoc插件与最新版本不兼容

## 解决方案

### 1. 安装必要插件
```bash
pip install pandoc-crossref
```

### 2. 修复后的最终命令
```bash
cd /Users/lqy/work/TabPFN/dissertation/latex/AI_Healthcare_Analytics_2025 && \
pandoc main.tex \
  --from=latex \
  --to=docx \
  --output="/Users/lqy/work/TabPFN/dissertation/word/AI_Healthcare_Analytics_2025/AI_Healthcare_Analytics_2025_final.docx" \
  --reference-doc="/Users/lqy/work/TabPFN/dissertation/word/AI_Healthcare_Analytics_2025/reference.docx" \
  --filter=pandoc-crossref \
  --metadata=link-citations:true \
  --metadata=link-bibliography:true \
  --list-of-figures \
  --list-of-tables \
  --bibliography=refs.bib \
  --csl=numeric.csl \
  --citeproc
```

## 关键改进点

1. **✅ 使用pandoc-crossref**：提供表格和图表自动编号功能
2. **✅ 启用list-of-tables**：确保生成表格目录
3. **✅ 保留LaTeX原生表格结构**：使用表格的\caption{}和\label{}
4. **✅ 版本兼容性**：避免了不兼容的插件版本

## 验证结果
生成的文件：`AI_Healthcare_Analytics_2025_final.docx`

现在应该正确显示：
- Table 1: The training (Cohort A) and testing (Cohort B) cohorts.
- 其他表格的正确编号
- 图表目录中的Figure编号

## 使用方法
直接运行修复后的命令，或使用提供的脚本：
```bash
cd /Users/lqy/work/TabPFN/dissertation/latex/AI_Healthcare_Analytics_2025
./final_convert_command.sh
```

## 注意事项
- 确保LaTeX源文件中的表格有正确的\caption{}和\label{}定义
- pandoc-crossref会自动处理表格编号
- 如需进一步定制，可参考pandoc-crossref文档