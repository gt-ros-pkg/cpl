


function grammar = load_grammar(file)



grammar.starting    = 1;
grammar.rules       = struct([]);
grammar.symbols     = struct([]);

fid = fopen(file);
assert(fid > 0);

while ~feof(fid)
    
    % read line
    s = fgetl(fid);
    if length(s) < 3, continue; end;
    disp(['read line:  ' s]);
    
    % parse
    %tokens = regexp(s, '\s', 'split');
    %tokens = strread(s,'%s','delimiter',' ');
    tokens = textscan(s, '%s');
    tokens = tokens{1};
    
    if strcmp(tokens{1}, '%')
        continue;
    end
    
    % get left
    left_id = [];
    try
        left_id = find(strcmp({grammar.symbols.name}, tokens{1}));
    end;
    if isempty(left_id)
        left_id = length(grammar.symbols) + 1;
        grammar.symbols(left_id).name = tokens{1};
    end
    grammar.symbols(left_id).is_terminal = 0;
    
    % check for terminal info line
    if ~strcmp(tokens{2}, '>')
        grammar.symbols(left_id).is_terminal = 1;
        grammar.symbols(left_id).detector_id = str2num(tokens{2});
        
        if length(tokens) == 4
            grammar.symbols(left_id).manual_params.duration_mean = str2num(tokens{3});
            grammar.symbols(left_id).manual_params.duration_var  = str2num(tokens{4});
            
        end
        
        continue;
    end
    
    % get right
    right_ids = [];
    for k=3:2:length(tokens)
        rid = find(strcmp({grammar.symbols.name}, tokens{k}));
        if isempty(rid)
            rid = length(grammar.symbols) + 1;
            grammar.symbols(rid).name = tokens{k};
            grammar.symbols(rid).is_terminal = 1;
        end
        right_ids(end+1) = rid;
    end
    
    % rule
    grammar.rules(end+1).id = length(grammar.rules) + 1;
    
    grammar.rules(end).left = left_id;
    grammar.rules(end).right = right_ids;
    grammar.rules(end).or_rule = 0;
    if length(tokens) >= 4 && strcmp(tokens{4}, 'or')
        n = (length(tokens) - 1) / 2;
        grammar.rules(end).or_rule = 1;
        grammar.rules(end).or_prob = ones(1, n) / n; 
    end
    
    grammar.symbols(left_id).rule_id = grammar.rules(end).id;
end


fclose(fid);

%% print grammar
disp '------------ print grammar rules'
for i=1:length(grammar.rules)
    
    r = '';
    r = [r grammar.symbols(grammar.rules(i).left).name];
    r = [r ' >>> '];
    
    for j=grammar.rules(i).right
        r = [r grammar.symbols(j).name];
        
        if grammar.rules(i).or_rule
            r = [r ' | '];
        else
            r = [r '  '];
        end
    end
    
    disp(r);
end



end







































