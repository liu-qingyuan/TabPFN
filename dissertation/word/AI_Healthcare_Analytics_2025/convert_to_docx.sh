#!/bin/bash
set -euo pipefail

cd "/Users/lqy/work/TabPFN/dissertation/latex/AI_Healthcare_Analytics_2025"

/opt/homebrew/bin/pandoc main.tex \
  -s \
  --from=latex \
  --to=docx \
  --output=/Users/lqy/work/TabPFN/dissertation/word/AI_Healthcare_Analytics_2025/AI_Healthcare_Analytics_2025.docx \
  --reference-doc=/Users/lqy/work/TabPFN/dissertation/word/AI_Healthcare_Analytics_2025/reference.docx \
  --lua-filter=fix_latex_tables.lua \
  --filter=pandoc-crossref \
  --lua-filter=latex_toc_to_word.lua \
  --metadata-file=crossref.yaml \
  --citeproc \
  --bibliography=refs.bib \
  --csl=numeric.csl \
  --metadata link-citations=true \
  --metadata link-bibliography=true \
  --list-of-figures \
  --list-of-tables
