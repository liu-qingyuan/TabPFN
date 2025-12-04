-- Normalize LaTeX width specs like \linewidth or 0.95\textwidth to percentages
-- that the docx writer understands.

local function convert_width(width)
  if not width then
    return nil
  end
  local factor, unit = width:match("^%s*([%d%.]*)\\(%a+)%s*$")
  if not unit then
    return nil
  end
  if unit == "linewidth" or unit == "textwidth" or unit == "columnwidth" then
    local num = tonumber(factor)
    if not num then
      num = 1
    end
    local pct = string.format("%.0f%%", num * 100)
    return pct
  end
  return nil
end

function Image(el)
  local new_width = convert_width(el.attributes["width"])
  if new_width then
    el.attributes["width"] = new_width
  end
  return el
end
