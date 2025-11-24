#!/bin/bash

# Final working command for LaTeX to DOCX conversion with table numbering
echo "Starting LaTeX to DOCX conversion with table numbering..."
echo "Note: This version focuses on pandoc-crossref functionality"

cd "/Users/lqy/work/TabPFN/dissertation/latex/AI_Healthcare_Analytics_2025"

# The working command that successfully converts with proper table numbering
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

if [ $? -eq 0 ]; then
    echo "✅ Conversion successful!"
    echo "📄 Output file: AI_Healthcare_Analytics_2025_final.docx"
    echo ""
    echo "🔍 Key improvements made:"
    echo "   1. ✓ Installed pandoc-crossref for table and figure numbering"
    echo "   2. ✓ Used LaTeX's native table captions and labels"
    echo "   3. ✓ Enabled list-of-tables and list-of-figures"
    echo "   4. ✓ Fixed pandoc version compatibility issues"
    echo ""
    echo "📋 This should now show:"
    echo "   - Table 1: The training (Cohort A) and testing (Cohort B) cohorts."
    echo "   - Proper figure numbering in the List of Figures"
    echo ""
else
    echo "❌ Conversion failed. Please check the error messages above."
    exit 1
fi