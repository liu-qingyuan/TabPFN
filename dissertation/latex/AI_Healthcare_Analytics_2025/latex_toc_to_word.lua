function Para(el)
  local text = pandoc.utils.stringify(el)
  
  if text == "INSERT_TOC_HERE" then
    return pandoc.RawBlock('openxml', [[
<w:p>
  <w:pPr>
    <w:pStyle w:val="TOCHeading"/>
  </w:pPr>
  <w:r>
    <w:t>TABLE OF CONTENTS</w:t>
  </w:r>
</w:p>
<w:p>
  <w:fldSimple w:instr="TOC \o &quot;1-3&quot; \h \z \u">
    <w:r>
      <w:rPr>
        <w:b/>
        <w:bCs/>
        <w:noProof/>
      </w:rPr>
      <w:t>Right-click to update field.</w:t>
    </w:r>
  </w:fldSimple>
</w:p>
]])
  end
  -- Removed LOF and LOT handling to rely on native Pandoc flags
end
