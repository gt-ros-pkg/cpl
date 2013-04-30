function s = nx_assign_struct(s, i, v)
%NX_ASSIGN_STRUCT Summary of this function goes here
%   Detailed explanation goes here

    s    = nx_extend_struct_fields(s, v);
    v    = nx_extend_struct_fields(v, s);
    s(i) = v;
    
    
end

