#!/bin/bash

# Convert LaTeX to DOCX with proper table numbering
# This script addresses the issue where table numbers (Table 1, Table 2) were not showing in the list of tables

cd "/Users/lqy/work/TabPFN/dissertation/latex/AI_Healthcare_Analytics_2025"

# Method 1: Direct LaTeX conversion with native numbering
echo "Method 1: Converting with native numbering support..."
pandoc main.tex \
  --from=latex+native_numbering \
  --to=docx \
  --output="/Users/lqy/work/TabPFN/dissertation/word/AI_Healthcare_Analytics_2025/AI_Healthcare_Analytics_method1.docx" \
  --reference-doc="/Users/lqy/work/TabPFN/dissertation/word/AI_Healthcare_Analytics_2025/reference.docx" \
  --filter=pandoc-crossref \
  --filter=pandoc-tablenos \
  --filter=pandoc-fignos \
  --metadata=number-tables:true \
  --metadata=number-figures:true \
  --metadata=link-citations:true \
  --metadata=link-bibliography:true \
  --list-of-figures \
  --list-of-tables \
  --bibliography=refs.bib \
  --csl=numeric.csl

if [ $? -eq 0 ]; then
    echo "Method 1 completed successfully"
else
    echo "Method 1 failed"
fi

# Method 2: Step-by-step conversion with intermediate Markdown
echo "Method 2: Converting via intermediate Markdown..."
pandoc main.tex \
  --from=latex \
  --to=markdown+pipe_tables+citations \
  --output=intermediate.md \
  --filter=pandoc-crossref \
  --bibliography=refs.bib \
  --csl=numeric.csl

if [ $? -eq 0 ]; then
    echo "Markdown conversion successful, now converting to DOCX..."
    pandoc intermediate.md \
      --from=markdown+pipe_tables+citations+native_numbering \
      --to=docx \
      --output="/Users/lqy/work/TabPFN/dissertation/word/AI_Healthcare_Analytics_2025/AI_Healthcare_Analytics_method2.docx" \
      --reference-doc="/Users/lqy/work/TabPFN/dissertation/word/AI_Healthcare_Analytics_2025/reference.docx" \
      --filter=pandoc-tablenos \
      --filter=pandoc-fignos \
      --metadata=number-tables:true \
      --metadata=number-figures:true \
      --metadata=link-citations:true \
      --metadata=link-bibliography:true \
      --list-of-figures \
      --list-of-tables

    if [ $? -eq 0 ]; then
        echo "Method 2 completed successfully"
    else
        echo "Method 2 failed at DOCX conversion"
    fi
else
    echo "Method 2 failed at Markdown conversion"
fi

# Method 3: Custom YAML metadata approach
echo "Method 3: Converting with custom metadata..."
pandoc main.tex \
  --from=latex+native_numbering \
  --to=docx \
  --output="/Users/lqy/work/TabPFN/dissertation/word/AI_Healthcare_Analytics_2025/AI_Healthcare_Analytics_method3.docx" \
  --reference-doc="/Users/lqy/work/TabPFN/dissertation/word/AI_Healthcare_Analytics_2025/reference.docx" \
  --metadata-file=metadata.yaml \
  --filter=pandoc-crossref \
  --filter=pandoc-tablenos \
  --filter=pandoc-fignos \
  --citeproc \
  --list-of-figures \
  --list-of-tables

if [ $? -eq 0 ]; then
    echo "Method 3 completed successfully"
else
    echo "Method 3 failed"
fi

echo "Conversion completed. Please check the generated DOCX files:"
echo "1. AI_Healthcare_Analytics_method1.docx"
echo "2. AI_Healthcare_Analytics_method2.docx"
echo "3. AI_Healthcare_Analytics_method3.docx"
echo ""
echo "Check which method successfully shows 'Table 1', 'Table 2' in the List of Tables."

# Clean up intermediate files
rm -f intermediate.md