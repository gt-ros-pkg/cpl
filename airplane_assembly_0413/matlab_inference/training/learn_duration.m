
disp ==========================================================
disp 'train action duration'
disp ==========================================================

grammar.symbols(1).duration_data = [];

for i=1:length(label)
    
    l = label(i);
    
    if strcmp(l.name, 'start') || strcmp(l.name, 'end')
        continue;
    end

    symbolid = actionname2symbolid(l.name, grammar);
    
    grammar.symbols(symbolid).duration_data(:,end+1) = l.end - l.start + 1;

end

for i=1:length(grammar.symbols)
    if ~isempty(grammar.symbols(i).duration_data)
        
        grammar.symbols(i).learntparams.duration_mean = mean(grammar.symbols(i).duration_data);
        grammar.symbols(i).learntparams.duration_var  = var(grammar.symbols(i).duration_data);
        
        disp(['Train duration for action ' grammar.symbols(i).name]);
        disp data
        disp(grammar.symbols(i).duration_data);
        disp mean
        disp(grammar.symbols(i).learntparams.duration_mean);
        disp var
        disp(grammar.symbols(i).learntparams.duration_var);
    end
end