function [ output_args ] = find_id_by_name( name, m, data )
%FIND_ID_BY_NAME Summary of this function goes here
%   Detailed explanation goes here


    for i=1:length(m.g)
        if strcmp(data.grammar.symbols(m.g(i).id).name, name),
            disp(i);
        end
    end


end

