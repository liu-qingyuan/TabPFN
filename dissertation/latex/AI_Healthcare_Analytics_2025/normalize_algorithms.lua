local algo_counter = 0
local label_to_num = {}

-- Resolve or assign a number for a given algorithm label
local function number_for_label(label)
  if label_to_num[label] then
    return label_to_num[label]
  end
  algo_counter = algo_counter + 1
  label_to_num[label] = algo_counter
  return algo_counter
end

-- Normalize inline references so Word shows "Algorithm N" even if crossref fails.
function RawInline(el)
  if el.format ~= "latex" then
    return nil
  end

  local text = el.text
  -- Equations: map eq labels to pandoc-crossref form (@eq:label)
  text = text:gsub("Eq~?%s*\\~?%s*\\ref%s*{%s*eq:([^}]+)%s*}", "Eq.~@eq:%1")
  text = text:gsub("\\eqref%s*{%s*eq:([^}]+)%s*}", "@eq:%1")
  text = text:gsub("\\ref%s*{%s*eq:([^}]+)%s*}", "@eq:%1")

  -- Algorithms: \ref{alg:foo} -> Algorithm N
  text = text:gsub("\\ref%s*{%s*(alg:[^}]+)%s*}", function(label)
    local num = number_for_label(label)
    return "Algorithm " .. tostring(num)
  end)

  if text ~= el.text then
    return pandoc.RawInline("latex", text)
  end
end

-- Convert LaTeX algorithm environments into a heading + code block
function RawBlock(el)
  if el.format ~= "latex" then
    return nil
  end

  if not el.text:match("\\begin{algorithm}") then
    return nil
  end

  local label = el.text:match("\\label%s*{%s*([^}]+)%s*}") or ""
  local caption = el.text:match("\\caption%s*{%s*(.-)%s*}") or "Algorithm"

  -- Extract body of algorithmic environment
  local body = el.text:match("\\begin{algorithmic}[^\n]*\n(.-)\n\\end{algorithmic}")
  if not body then
    body = el.text:match("\\begin{algorithmic}[^\n]*%s*(.-)\\end{algorithmic}")
  end
  body = body or ""

  -- Heuristic expansion of algorithmic commands to readable lines
  local lines = {}
  for line in body:gmatch("[^\n]+") do
    local trimmed = line:gsub("^%s+", "")
    trimmed = trimmed:gsub("\\COMMENT%s*{(.-)}", "%(%1%)")

    trimmed = trimmed:gsub("^\\REQUIRE%s*(.*)", "Input: %1")
    trimmed = trimmed:gsub("^\\ENSURE%s*(.*)", "Output: %1")
    trimmed = trimmed:gsub("^\\STATE%s*(.*)", "- %1")
    trimmed = trimmed:gsub("^\\item%s*(.*)", "- %1")
    trimmed = trimmed:gsub("^\\FOR%s*{?(.-)}?", "For %1:")
    trimmed = trimmed:gsub("^\\IF%s*{?(.-)}?", "If %1:")
    trimmed = trimmed:gsub("^\\ELSEIF%s*{?(.-)}?", "Else if %1:")
    trimmed = trimmed:gsub("^\\ELSE%s*$", "Else:")
    trimmed = trimmed:gsub("^\\WHILE%s*{?(.-)}?", "While %1:")
    trimmed = trimmed:gsub("^\\RETURN%s*(.*)", "Return %1")
    trimmed = trimmed:gsub("^\\END%u+", "End")

    trimmed = trimmed:gsub("\\\\%s*$", "")
    trimmed = trimmed:gsub("%s*;?%s*$", "")

    if trimmed ~= "" then
      table.insert(lines, trimmed)
    end
  end

  local normalized_body = table.concat(lines, "\n")

  -- Determine number
  local num = label ~= "" and number_for_label(label) or number_for_label("alg:auto" .. algo_counter + 1)

  -- Build heading paragraph and code block
  local heading = pandoc.Para({
    pandoc.Str("Algorithm " .. tostring(num) .. ": " .. caption)
  })
  local cb_attr = pandoc.Attr(label ~= "" and label or ("alg:auto" .. tostring(num)), {"listing"})
  local codeblock = pandoc.CodeBlock(normalized_body, cb_attr)

  return {heading, codeblock}
end
