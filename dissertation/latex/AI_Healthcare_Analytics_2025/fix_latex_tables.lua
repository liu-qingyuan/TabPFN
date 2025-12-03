function Div(el)
  -- Identify Divs created by Pandoc for LaTeX tables with \label{tab:xxx}
  if el.identifier:match("^tab:") and #el.content == 1 and el.content[1].t == "Table" then
    local tbl = el.content[1]
    -- Rename tab: to tbl: for pandoc-crossref compatibility
    -- and move the identifier from the wrapper Div to the Table itself
    tbl.identifier = el.identifier:gsub("^tab:", "tbl:")
    return tbl
  end
end

function Cite(el)
  -- Update citation IDs in text (e.g. [@tab:foo])
  for _, citation in ipairs(el.citations) do
    if citation.citationId and citation.citationId:match("^tab:") then
      citation.citationId = citation.citationId:gsub("^tab:", "tbl:")
    end
  end
  return el
end

function Link(el)
  -- Update internal links (e.g. \ref{tab:foo} -> Link)
  if el.attributes and el.attributes['reference'] and el.attributes['reference']:match("^tab:") then
    el.attributes['reference'] = el.attributes['reference']:gsub("^tab:", "tbl:")
  end
  
  -- Update target anchors (e.g. #tab:foo)
  if el.target:match("^#tab:") then
      el.target = el.target:gsub("^#tab:", "#tbl:")
  end
  return el
end

function Inlines(inlines)
  local out = {}
  local i = 1
  while i <= #inlines do
    local cur = inlines[i]
    local nxt = inlines[i + 1]

    -- Pattern: Str "Table ef" + Span {tab:foo}
    if cur.t == "Str" and nxt and nxt.t == "Span" then
      local span_text = pandoc.utils.stringify(nxt)
      if span_text and (span_text:match("^tab:") or span_text:match("^tbl:")) then
        local label = span_text:gsub("^tab:", "tbl:")
        local head = cur.c or ""
        head = head:gsub("ef", "")
        table.insert(out, pandoc.Str(head))
        table.insert(out, pandoc.Str("@" .. label))
        i = i + 2
        goto continue
      end
    end

    -- Standalone Span with tab:/tbl: (e.g., produced without the Str "ef")
    if cur.t == "Span" then
      local span_text = pandoc.utils.stringify(cur)
      if span_text and (span_text:match("^tab:") or span_text:match("^tbl:")) then
        local label = span_text:gsub("^tab:", "tbl:")
        -- If previous output ends with "ef", trim it
        if #out > 0 and out[#out].t == "Str" then
          out[#out].c = out[#out].c:gsub("ef", "")
        end
        table.insert(out, pandoc.Str("@" .. label))
        i = i + 1
        goto continue
      end
    end

    table.insert(out, cur)
    i = i + 1
    ::continue::
  end
  return out
end

function Table(el)
  -- Normalize table identifiers from tab: to tbl: for crossref
  if el.identifier and el.identifier:match("^tab:") then
    el.identifier = el.identifier:gsub("^tab:", "tbl:")
  end
  return el
end
