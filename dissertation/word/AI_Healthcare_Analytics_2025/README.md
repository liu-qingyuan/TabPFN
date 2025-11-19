# LaTeX 转 Word 完整指南

## 📋 项目概述

本指南说明如何将 LaTeX 论文转换为 Word 文档，支持数字引用格式、可点击跳转、自定义模板等功能。

## 📁 目录结构

```
/Users/lqy/work/TabPFN/dissertation/
├── latex/AI_Healthcare_Analytics_2025/
│   ├── main.tex                    # 主LaTeX文件
│   ├── refs.bib                    # 参考文献数据库
│   ├── Section/                    # 论文章节
│   │   ├── Introduction.tex
│   │   ├── Methods.tex
│   │   ├── Results.tex
│   │   └── ...
│   └── numeric.csl                 # 数字引用样式
└── word/AI_Healthcare_Analytics_2025/
    ├── AI_Healthcare_Analytics_2025_with_template.docx  # 最终Word文档
    ├── reference.docx                                        # Word模板
    └── README.md                                            # 本文档
```

## 🛠️ 环境准备

### 安装必要的工具

```bash
# macOS (使用 Homebrew)
brew install pandoc
brew install pandoc-crossref

# Ubuntu/Debian
sudo apt-get install pandoc
# pandoc-crossref 需要手动下载安装

# Windows (使用 Chocolatey)
choco install pandoc
choco install pandoc-crossref
```

### 验证安装

```bash
pandoc --version
pandoc-crossref --version
```

## 🎯 核心转换命令

### 完整命令（推荐）

**方法一：一键运行（推荐）**

```bash

```

**方法二：分步执行**

```bash
# 1. 切换到LaTeX源目录
cd /Users/lqy/work/TabPFN/dissertation/latex/AI_Healthcare_Analytics_2025

# 2. 运行转换命令
pandoc main.tex \
  --from=latex \
  --to=docx \
  --output=/Users/lqy/work/TabPFN/dissertation/word/AI_Healthcare_Analytics_2025/AI_Healthcare_Analytics_2025_with_template.docx \
  --reference-doc=/Users/lqy/work/TabPFN/dissertation/word/AI_Healthcare_Analytics_2025/reference.docx \
  --filter=pandoc-crossref \
  --citeproc \
  --bibliography=refs.bib \
  --csl=numeric.csl \
  --metadata link-citations=true \
  --metadata link-bibliography=true
```

**说明**：

- **工作目录**: 必须在LaTeX源文件目录运行（包含 `main.tex`, `refs.bib`, `numeric.csl`）
- **输出位置**: Word文件保存到 `/Users/lqy/work/TabPFN/dissertation/word/AI_Healthcare_Analytics_2025/`
- **模板文件**: 使用word目录中的 `reference.docx`

### 参数详解

| 参数                                  | 作用                     | 必需    |
| ------------------------------------- | ------------------------ | ------- |
| `--from=latex`                      | 输入文件格式为 LaTeX     | ✅      |
| `--to=docx`                         | 输出格式为 Word          | ✅      |
| `--output=文件名.docx`              | 指定输出文件名           | ✅      |
| `--reference-doc=reference.docx`    | 使用自定义Word模板       | ⭐ 推荐 |
| `--filter=pandoc-crossref`          | 处理图表公式交叉引用     | ✅      |
| `--citeproc`                        | 启用引用处理器           | ✅      |
| `--bibliography=refs.bib`           | 指定参考文献数据库       | ✅      |
| `--csl=numeric.csl`                 | 指定引用样式（数字格式） | ✅      |
| `--metadata link-citations=true`    | 使引用可点击跳转         | ⭐ 重要 |
| `--metadata link-bibliography=true` | 参考文献中链接可点击     | ⭐ 推荐 |

## 📝 引用样式配置

### 数字引用格式（CSL文件）

创建 `numeric.csl` 文件：

```xml
<?xml version="1.0" encoding="utf-8"?>
<style xmlns="http://purl.org/net/xbiblio/csl" class="in-text" version="1.0">
  <info>
    <title>Numeric</title>
    <id>http://www.zotero.org/styles/numeric</id>
    <updated>2024-01-01T00:00:00+00:00</updated>
    <summary>Numeric citation style</summary>
    <category citation-format="numeric"/>
    <category field="generic-base"/>
  </info>

  <!-- 引用格式：[1], [2,3] -->
  <citation collapse="citation-number">
    <sort>
      <key variable="citation-number"/>
    </sort>
    <layout delimiter="," prefix="[" suffix="]">
      <text variable="citation-number"/>
    </layout>
  </citation>

  <!-- 参考文献列表格式 -->
  <bibliography>
    <sort>
      <key variable="citation-number"/>
    </sort>
    <layout>
      <text variable="citation-number" prefix="[" suffix="]"/>
      <text macro="author" suffix=" "/>
      <text macro="year" suffix=". "/>
      <text macro="title" suffix=" "/>
      <text macro="journal"/>
    </layout>
  </bibliography>
</style>
```

### 其他常用CSL样式

```bash
# 下载不同期刊的CSL样式
curl -o nature.csl https://raw.githubusercontent.com/citation-style-language/styles/master/nature.csl
curl -o ieee.csl https://raw.githubusercontent.com/citation-style-language/styles/master/ieee.csl
curl -o vancouver.csl https://raw.githubusercontent.com/citation-style-language/styles/master/vancouver.csl
```

## 🎨 自定义Word模板

### 生成默认模板

```bash
cd /Users/lqy/work/TabPFN/dissertation/word/AI_Healthcare_Analytics_2025
pandoc -o custom-reference.docx --print-default-data-file reference.docx
```

**注意**：正确的模板生成命令是 `pandoc -o custom-reference.docx --print-default-data-file reference.docx`，生成的模板文件约 11KB。

### 自定义模板步骤

1. **打开 `reference.docx`**
2. **修改样式**：
   - 标题1、标题2、标题3 的字体和字号
   - 正文字体（推荐 Times New Roman 12pt）
   - 表格样式
   - 页面边距
3. **保存模板**
4. **在转换命令中使用 `--reference-doc=reference.docx`**

### 常用样式推荐

```markdown
## 推荐格式：
- **标题**: Times New Roman, 16pt, 加粗
- **一级标题**: Times New Roman, 14pt, 加粗
- **二级标题**: Times New Roman, 12pt, 加粗
- **正文**: Times New Roman, 12pt
- **参考文献**: Times New Roman, 10pt
- **行距**: 1.5倍
- **页边距**: 上下2.54cm，左右3.17cm
```

## 🔧 LaTeX文件修改

### 添加References标题

在 `main.tex` 文件末尾添加：

```latex
% 参考文献
\section*{References}
\bibliographystyle{unsrt}
\bibliography{refs}
```

### 确保引用格式正确

LaTeX中使用的引用格式：

```latex
% 单个引用
This was shown by \cite{author2020}.

% 多个引用
Several studies \cite{author2020,smith2019,jones2018}.

% 图表引用
As shown in Figure~\ref{fig:results}...
```

## ⚡ 快速命令

### 基础转换（无模板）

```bash
pandoc main.tex \
  --from=latex \
  --to=docx \
  --output=paper.docx \
  --filter=pandoc-crossref \
  --citeproc \
  --bibliography=refs.bib \
  --csl=numeric.csl \
  --metadata link-citations=true
```

### 完整转换（带模板）

```bash
pandoc main.tex \
  --from=latex \
  --to=docx \
  --output=paper.docx \
  --reference-doc=reference.docx \
  --filter=pandoc-crossref \
  --citeproc \
  --bibliography=refs.bib \
  --csl=numeric.csl \
  --metadata link-citations=true \
  --metadata link-bibliography=true
```

### 不同引用格式

```bash
# APA格式 (Author, Year)
pandoc main.tex \
  --output=paper_apa.docx \
  --csl=apa.csl \
  --metadata link-citations=true

# Nature格式
pandoc main.tex \
  --output=paper_nature.docx \
  --csl=nature.csl \
  --metadata link-citations=true
```

## ⚠️ 常见问题

### 1. 引用不可点击

**原因**: 缺少 `--metadata link-citations=true`

**解决**: 添加该参数到pandoc命令中

### 2. 参考文献没有标题

**原因**: LaTeX中缺少 `\section*{References}`

**解决**: 在main.tex中添加References标题

### 3. 引用格式错误

**原因**: CSL文件格式不正确或路径错误

**解决**: 检查CSL文件语法和路径

### 4. 图片无法显示

**原因**: 图片路径不正确或格式不支持

**解决**:

- 使用相对路径
- 转换图片为PNG/JPG格式
- 确保图片文件存在

### 5. 交叉引用编号错误

**原因**: 缺少 `--filter=pandoc-crossref`

**解决**: 添加该参数并确保LaTeX中有正确的 `\label{}`和 `\ref{}`

### 6. 模板字体样式不生效

**可能原因**:

- 使用了错误的模板生成命令
- 模板文件路径不正确
- 默认模板只包含基本样式

**解决方案**:

```bash
# 生成正确的模板文件
pandoc -o custom-reference.docx --print-default-data-file reference.docx

# 在Word中打开模板文件，手动修改样式后保存
open custom-reference.docx

# 使用正确的模板重新生成
pandoc main.tex --reference-doc=custom-reference.docx [其他参数...]
```

### 7. 工作目录错误

**错误**：在word目录运行转换命令

**解决**：必须在LaTeX源文件目录运行：

```bash
cd /Users/lqy/work/TabPFN/dissertation/latex/AI_Healthcare_Analytics_2025
# 然后运行pandoc命令
```

## 📚 参考资料

- [Pandoc官方文档](https://pandoc.org/)
- [pandoc-crossref GitHub](https://github.com/lierdakil/pandoc-crossref)
- [CSL样式库](https://github.com/citation-style-language/styles)
- [Citation Style Language官网](https://citationstyles.org/)

## 🎉 总结

使用本指南，你可以：

1. ✅ 将LaTeX完美转换为Word
2. ✅ 获得数字引用格式 `[1]`, `[2,3]`
3. ✅ 实现点击引用跳转到参考文献
4. ✅ 自定义Word模板样式
5. ✅ 处理图表公式的交叉引用

最终生成的Word文档完全符合学术出版要求！
