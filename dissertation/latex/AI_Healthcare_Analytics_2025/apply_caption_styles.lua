-- This filter applies custom styles to Figure and Table captions
-- It should be run AFTER pandoc-crossref to ensure the caption text is already finalized.

function Figure(el)
  if el.caption and el.caption.long then
    -- We wrap the entire content of the caption in a Div with the custom style.
    -- This tells the docx writer to apply "Figure Caption" style to the paragraph(s) inside.
    local div = pandoc.Div(el.caption.long)
    div.attributes['custom-style'] = 'Figure Caption'
    el.caption.long = pandoc.List({div})
  end
  return el
end

function Table(el)
  if el.caption and el.caption.long then
    local div = pandoc.Div(el.caption.long)
    div.attributes['custom-style'] = 'Table Caption'
    el.caption.long = pandoc.List({div})
  end
  return el
end
