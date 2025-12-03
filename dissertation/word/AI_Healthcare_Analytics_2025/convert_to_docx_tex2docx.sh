#!/bin/bash
set -euo pipefail

# Use the tex2docx panflute filter to convert the LaTeX thesis to DOCX with
# cross-references (eq/table/figure/section) resolved.

LATEX_DIR="/Users/lqy/work/TabPFN/dissertation/latex/AI_Healthcare_Analytics_2025"
WORD_DIR="/Users/lqy/work/TabPFN/dissertation/word/AI_Healthcare_Analytics_2025"
FILTER="/Users/lqy/work/TabPFN/dissertation/word/tex2docx/filter.py"
INCLUDE_FILTER="/Users/lqy/work/TabPFN/dissertation/word/tex2docx/include_files.lua"

INPUT_TEX="$LATEX_DIR/main.tex"
OUTPUT_DOCX="$WORD_DIR/AI_Healthcare_Analytics_2025.docx"
REF_DOC="$WORD_DIR/reference.docx"
BIBLIO="$LATEX_DIR/refs.bib"
CSL_STYLE="$LATEX_DIR/numeric.csl"

# NOTE: Ensure the LaTeX project is up to date (pdflatex/bibtex) so labels exist
# before running this script if the filter relies on .aux data.

/opt/homebrew/bin/pandoc "$INPUT_TEX" \
  --from=latex+raw_tex \
  --to=docx \
  --citeproc \
  --bibliography "$BIBLIO" \
  --csl "$CSL_STYLE" \
  --resource-path "$LATEX_DIR" \
  --reference-doc "$REF_DOC" \
  --lua-filter "$INCLUDE_FILTER" \
  -F "$FILTER" \
  --output "$OUTPUT_DOCX"

echo "DOCX generated at: $OUTPUT_DOCX"
