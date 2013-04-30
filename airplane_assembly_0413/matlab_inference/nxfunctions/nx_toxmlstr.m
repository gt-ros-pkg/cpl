function s = nx_toxmlstr(v, tag)
%TOXMLSTR Summary of this function goes here
%   Detailed explanation goes here
    if ~exist('tag')
        tag = 'data';
    end
    
    if ischar(v)
        s = ['<' tag '>' v '</' tag '>'];
        
    elseif isnumeric(v)
        s = ['<' tag ' rows="' num2str(size(v,1)) '" cols="' num2str(size(v,2)) '">' num2str(v(:)') '</' tag '>'];
        
    elseif isstruct(v)
        
        if length(v) == 1
            s = ['<' tag '>'];

            fields = fieldnames(v);
            for i=1:length(fields)
                s = [s nx_toxmlstr(v.(fields{i}), fields{i})];
            end

            s = [s '</' tag '>'];
        else
            s = '';
            
            for i=1:length(v)
                s = [s nx_toxmlstr(v(i), tag)];
            end
        end
    end

end

