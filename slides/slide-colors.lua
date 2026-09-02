function Header(el)
  if el.level == 2 then
    el.attributes["data-background-color"] = "#ab82c5"
  elseif el.level == 3 then
    el.attributes["data-background-color"] = "#f2f2f2"
  end
  return el
end
