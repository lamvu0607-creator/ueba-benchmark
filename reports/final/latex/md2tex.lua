-- Pandoc Lua filter: markdown report -> LaTeX
-- * relative repo links become plain text (they mean nothing inside a PDF)
-- * image paths are made relative to this folder
-- * inline code may break after underscores / slashes
-- * pipe tables get column widths from their content

local function tex_escape(s)
  s = s:gsub('\\', '\\textbackslash{}')
  s = s:gsub('([#$%%&{}])', '\\%1')
  s = s:gsub('~', '\\textasciitilde{}')
  s = s:gsub('%^', '\\textasciicircum{}')
  return s
end

local function Link(el)
  if el.target:match('^https?://') then return el end
  return el.content
end

local function Image(el)
  if not el.src:match('^https?://') and not el.src:match('^/') then
    el.src = '../' .. el.src
  end
  return el
end

local function Code(el)
  local s = tex_escape(el.text)
  s = s:gsub('_', '\\_\\allowbreak{}')
  s = s:gsub('/', '/\\allowbreak{}')
  s = s:gsub('%.', '.\\allowbreak{}')
  s = s:gsub('([(=,])', '%1\\allowbreak{}')
  return pandoc.RawInline('latex', '\\texttt{' .. s .. '}')
end

local function Str(el)
  if el.text:find('_') and utf8.len(el.text) > 6 then
    local s = tex_escape(el.text):gsub('_', '\\_\\allowbreak{}')
    return pandoc.RawInline('latex', s)
  end
end

local function CodeBlock(el)
  return pandoc.RawBlock('latex',
    '\\begin{Verbatim}[breaklines,breakanywhere,fontsize=\\footnotesize]\n' .. el.text .. '\n\\end{Verbatim}')
end

local function Table(tbl)
  local ncol = #tbl.colspecs
  local maxlen = {}
  for i = 1, ncol do maxlen[i] = 3 end
  local function scan(rows)
    for _, row in ipairs(rows) do
      for i, cell in ipairs(row.cells) do
        if i <= ncol then
          local n = utf8.len(pandoc.utils.stringify(cell.contents)) or 3
          if n > maxlen[i] then maxlen[i] = n end
        end
      end
    end
  end
  scan(tbl.head.rows)
  for _, b in ipairs(tbl.bodies) do scan(b.body) end
  local total = 0
  for i = 1, ncol do total = total + maxlen[i] end
  -- short tables: let LaTeX size the columns naturally
  if total + 3 * ncol <= 95 then
    for i = 1, ncol do tbl.colspecs[i] = {tbl.colspecs[i][1], nil} end
    return tbl
  end
  local w, sum = {}, 0
  for i = 1, ncol do
    w[i] = math.max(math.min(maxlen[i], 70), 9) ^ 0.8
    sum = sum + w[i]
  end
  for i = 1, ncol do
    tbl.colspecs[i] = {tbl.colspecs[i][1], 0.98 * w[i] / sum}
  end
  if ncol >= 6 then
    -- wide tables: smaller font so long identifiers fit
    return {pandoc.RawBlock('latex', '{\\footnotesize\\renewcommand{\\small}{\\footnotesize}'),
            tbl, pandoc.RawBlock('latex', '}')}
  end
  return tbl
end

-- Bold, as in the weekly reports: only a lead-in label at the start of a paragraph or list item
-- ("Lượng tiêm:", "Hình 4.1.") and the label column of tables. Every other bold is dropped.
local function is_label(strong)
  local t = pandoc.utils.stringify(strong)
  return t:match('[:.]$') ~= nil
end

local function mark_leadin(block)
  if (block.t == 'Para' or block.t == 'Plain') and #block.content > 0
      and block.content[1].t == 'Strong' and is_label(block.content[1]) then
    block.content[1] = pandoc.Span(block.content[1].content, {class = 'keepbold'})
  end
  return block
end

-- "**Bảng 4.1: ...**" right before a table becomes that table's caption, so it stays with the
-- table across page breaks; it is set centred and not bold, like the weekly reports.
local function attach_captions(doc)
  local out = {}
  local blocks = doc.blocks
  local i = 1
  while i <= #blocks do
    local b, nxt = blocks[i], blocks[i + 1]
    if b.t == 'Para' and nxt and nxt.t == 'Table' and #b.content >= 1 and b.content[1].t == 'Strong'
        and pandoc.utils.stringify(b.content[1]):match('^Bảng [%dA-Z]') then
      local inl = {}
      for _, x in ipairs(b.content[1].content) do table.insert(inl, x) end
      for k = 2, #b.content do table.insert(inl, b.content[k]) end
      nxt.caption = {long = {pandoc.Plain(inl)}}
      table.insert(out, nxt)
      i = i + 2
    else
      table.insert(out, b)
      i = i + 1
    end
  end
  doc.blocks = out
  return doc
end

local function first_column_bold(tbl)
  local function cell_bold(cell)
    local txt = pandoc.utils.stringify(cell.contents)
    if txt == '' or txt:match('^[%d%-+–−,.%%]') then return end
    local blocks = cell.contents
    if #blocks == 1 and (blocks[1].t == 'Plain' or blocks[1].t == 'Para') then
      blocks[1].content = {pandoc.Span(blocks[1].content, {class = 'keepbold'})}
    end
  end
  for _, b in ipairs(tbl.bodies) do
    for _, row in ipairs(b.body) do
      if #row.cells > 1 then cell_bold(row.cells[1]) end
    end
  end
  return tbl
end

local function Span(el)
  if el.classes:includes('keepbold') then return pandoc.Strong(el.content) end
end

-- Table widths must be measured before inline code is turned into raw LaTeX.
return {
  {Pandoc = attach_captions},
  {Para = mark_leadin, Plain = mark_leadin},
  {Strong = function(el) return el.content end},
  {Table = first_column_bold},
  {Span = Span},
  {Table = Table},
  {Link = Link, Image = Image, Code = Code, Str = Str, CodeBlock = CodeBlock},
}
