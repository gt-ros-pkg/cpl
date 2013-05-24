function s = get_symbol_by_name(grammar, name)
%GET_SYMBOL_BY_NAME Summary of this function goes here
%   Detailed explanation goes here

   s = grammar.symbols(actionname2symbolid(name, grammar));
   
end

