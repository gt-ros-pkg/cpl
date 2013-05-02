function symbolid = actionname2symbolid( actionname, grammar )
    
    symbolid = find(strcmp({grammar.symbols.name}, actionname));


end

