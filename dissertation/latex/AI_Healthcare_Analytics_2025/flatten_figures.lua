-- Flatten pandoc-crossref Figure blocks so docx writer doesn't wrap them
-- in narrow FigureTable layouts (which shrink images). We keep the figure
-- identifier for hyperlinks, render the image in a plain paragraph, and
-- place the numbered caption in a separate caption-style paragraph.

local function to_inlines(blocks)
  return pandoc.utils.blocks_to_inlines(blocks)
end

local function extract_image_para(blocks)
  for _, blk in ipairs(blocks) do
    if blk.t == "Para" or blk.t == "Plain" then
      for _, inline in ipairs(blk.content) do
        if inline.t == "Image" then
          return pandoc.Para({ inline })
        end
      end
    end
  end
  return nil
end

function Figure(fig)
  -- Take the first block as the image container.
  local img_block = fig.content[1]
  if not img_block then
    return nil
  end

  -- Build an image paragraph and wrap it in a styled Div.
  local img_para = extract_image_para(fig.content)
  if not img_para then
    return nil
  end
  local img_div = pandoc.Div({ img_para }, pandoc.Attr("", {}, { ["custom-style"] = "Figure" }))

  -- Build a caption paragraph using ImageCaption style.
  local caption_blocks = (fig.caption and fig.caption.long) or {}
  local caption_inlines = to_inlines(caption_blocks)
  local caption_para = pandoc.Para(caption_inlines)
  local caption_div = pandoc.Div({ caption_para }, pandoc.Attr("", {}, { ["custom-style"] = "Image Caption" }))

  -- Wrap both in a Div to retain the original identifier for hyperlinks.
  return pandoc.Div({ img_div, caption_div }, fig.attr)
end
