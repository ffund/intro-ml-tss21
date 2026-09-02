local format = FORMAT

function Pandoc(doc)
  local audience = 'combined'
  if doc.meta.audience then
    audience = pandoc.utils.stringify(doc.meta.audience)
  end

  return doc:walk {
    Header = function(el)
      if format ~= 'revealjs' and el.attributes['data-menu-title'] and #el.content == 0 then
        return {}
      end
      return el
    end,
    Div = function(el)
      if el.classes:includes('handout-only') and format == 'revealjs' then
        return {}
      end
      if el.classes:includes('slides-only') and format ~= 'revealjs' then
        return {}
      end
      if el.classes:includes('grad-only') and audience ~= 'grad' and audience ~= 'combined' then
        return {}
      end
      if el.classes:includes('undergrad-only') and audience ~= 'undergrad' and audience ~= 'ugrad' and audience ~= 'combined' then
        return {}
      end
      return el
    end
  }
end
