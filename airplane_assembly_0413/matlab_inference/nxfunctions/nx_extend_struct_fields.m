function s1 = nx_extend_struct_fields( s1, s2 )
%NX_EXTEND_STRUCT_FIELDS Summary of this function goes here
%   Detailed explanation goes here

    fields = fieldnames(s2);

    for i=1:numel(fields)
        
        if ~isfield(s1, fields{i})
            
            s1().(fields{i}) = [];
            
        end
    end
end

