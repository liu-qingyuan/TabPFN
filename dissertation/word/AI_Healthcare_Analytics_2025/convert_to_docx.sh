#!/bin/bash
set -euo pipefail

cd "/Users/lqy/work/TabPFN/dissertation/latex/AI_Healthcare_Analytics_2025"

# Flatten LaTeX inputs so pandoc-crossref sees a single document (avoids per-file counter resets)
tmp_tex=$(mktemp /tmp/ai_healthcare_expanded.XXXXXX)
trap 'rm -f "$tmp_tex"' EXIT
/opt/homebrew/bin/latexpand main.tex > "$tmp_tex"

/opt/homebrew/bin/pandoc "$tmp_tex" \
  -s \
  --from=latex+raw_tex \
  --to=docx \
  --output=/Users/lqy/work/TabPFN/dissertation/word/AI_Healthcare_Analytics_2025/AI_Healthcare_Analytics_2025.docx \
  --reference-doc=/Users/lqy/work/TabPFN/dissertation/word/AI_Healthcare_Analytics_2025/reference.docx \
  --lua-filter=/Users/lqy/work/TabPFN/dissertation/word/tex2docx/convert_refs.lua \
  --lua-filter=/Users/lqy/work/TabPFN/dissertation/word/tex2docx/map_math_macros.lua \
  --lua-filter=/Users/lqy/work/TabPFN/dissertation/latex/AI_Healthcare_Analytics_2025/fix_latex_tables.lua \
  --lua-filter=normalize_algorithms.lua \
  --filter=pandoc-crossref \
  --lua-filter=/Users/lqy/work/TabPFN/dissertation/latex/AI_Healthcare_Analytics_2025/latex_toc_to_word.lua \
  --metadata-file=crossref.yaml \
  --metadata chapters=false \
  --metadata linkReferences=true \
  --metadata crossrefYaml=/Users/lqy/work/TabPFN/dissertation/latex/AI_Healthcare_Analytics_2025/crossref.yaml \
  --metadata links-as-notes=false \
  --reference-location=block \
  --top-level-division=section \
  --number-sections \
  --citeproc \
  --bibliography=refs.bib \
  --csl=numeric.csl \
  --metadata link-citations=true \
  --metadata link-bibliography=true \
  --list-of-figures \
  --list-of-tables
