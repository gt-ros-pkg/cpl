function did = actionname2detectorid( actionname, grammar )

    did = [];
    
    for i=1:length(grammar.symbols)
        if strcmp(grammar.symbols(i).name, actionname)
            did = grammar.symbols(i).detector_id;
            return;
        end
    end
    
end

