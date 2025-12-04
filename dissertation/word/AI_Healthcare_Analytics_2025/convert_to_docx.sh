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
  --resource-path="$(pwd):$(pwd)/img" \
  --lua-filter=/Users/lqy/work/TabPFN/dissertation/word/tex2docx/convert_refs.lua \
  --lua-filter=/Users/lqy/work/TabPFN/dissertation/word/tex2docx/normalize_equations.lua \
  --lua-filter=/Users/lqy/work/TabPFN/dissertation/word/tex2docx/map_math_macros.lua \
  --lua-filter=/Users/lqy/work/TabPFN/dissertation/latex/AI_Healthcare_Analytics_2025/normalize_image_width.lua \
  --lua-filter=/Users/lqy/work/TabPFN/dissertation/latex/AI_Healthcare_Analytics_2025/fix_latex_tables.lua \
  --lua-filter=normalize_algorithms.lua \
  --filter=pandoc-crossref \
  --lua-filter=/Users/lqy/work/TabPFN/dissertation/latex/AI_Healthcare_Analytics_2025/flatten_figures.lua \
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

# Align the List of Figures field with the actual caption style used in the document
python - <<'PY'
import os
import zipfile
import xml.etree.ElementTree as ET

DOCX = "/Users/lqy/work/TabPFN/dissertation/word/AI_Healthcare_Analytics_2025/AI_Healthcare_Analytics_2025.docx"
TMP = DOCX + ".tmp"

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)

def qname(name: str) -> str:
    return f"{{{NS['w']}}}{name}"

with zipfile.ZipFile(DOCX, "r") as zin, zipfile.ZipFile(TMP, "w") as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename != "word/document.xml":
            zout.writestr(item, data)
            continue

        # Parse XML
        root = ET.fromstring(data)
        body = root.find("w:body", NS)
        if body is None:
            zout.writestr(item, data)
            continue

        # Find the List of Figures content control and replace it with a plain TOC field.
        replaced = False
        for idx, child in enumerate(list(body)):
            doc_part = child.find(".//w:docPartGallery", NS)
            if doc_part is not None and doc_part.get(qname("val")) == "List of Figures":
                # Build replacement paragraphs
                p_title = ET.Element(qname("p"))
                ppr = ET.SubElement(p_title, qname("pPr"))
                ET.SubElement(ppr, qname("pStyle"), {qname("val"): "TOC"})
                r = ET.SubElement(p_title, qname("r"))
                ET.SubElement(r, qname("t"), {"xml:space": "preserve"}).text = "List of Figures"

                p_field = ET.Element(qname("p"))
                ET.SubElement(p_field, qname("fldSimple"), {qname("instr"): r'TOC \h \z \t "Image Caption,1,ImageCaption,1"'})

                body.remove(child)
                body.insert(idx, p_field)
                body.insert(idx, p_title)
                replaced = True
                break

        xml_out = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        zout.writestr(item, xml_out)

os.replace(TMP, DOCX)
PY
