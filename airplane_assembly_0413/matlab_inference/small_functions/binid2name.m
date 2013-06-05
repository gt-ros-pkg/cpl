function name = binid2name( bin_id )
%BINID2NAME Summary of this function goes here
%   Detailed explanation goes here

    name = 'N/A';
    
    if bin_id == 3
        name = 'Body';
    elseif bin_id == 11
        name = 'Nose A';
    elseif bin_id == 10
        name = 'Nose H';
    elseif bin_id == 12
        name = 'Wing AT';
    elseif bin_id == 2
        name = 'Wing AD';
    elseif bin_id == 7
        name = 'Wing H';
    elseif bin_id == 14
        name = 'Tail AT';
    elseif bin_id == 15
        name = 'Tail AD';
    elseif bin_id == 13
        name = 'Tail H';
     
    end
end

