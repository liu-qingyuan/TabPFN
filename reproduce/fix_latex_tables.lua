function Div(el)
  if el.identifier:match("^tab:") and #el.content == 1 and el.content[1].t == "Table" then
    local tbl = el.content[1]
    -- Rename tab: to tbl: for pandoc-crossref compatibility
    tbl.identifier = el.identifier:gsub("^tab:", "tbl:")
    return tbl
  end
end

function Cite(el)
  for _, citation in ipairs(el.citations) do
    if citation.citationId:match("^tab:") then
      citation.citationId = citation.citationId:gsub("^tab:", "tbl:")
    end
  end
  return el
end

function Link(el)
  -- Handle LaTeX \ref conversion
  if el.attributes and el.attributes['reference'] and el.attributes['reference']:match("^tab:") then
    el.attributes['reference'] = el.attributes['reference']:gsub("^tab:", "tbl:")
  end
  
  -- Handle standard internal links
  if el.target:match("^#tab:") then
      el.target = el.target:gsub("^#tab:", "#tbl:")
  end
  return el
end