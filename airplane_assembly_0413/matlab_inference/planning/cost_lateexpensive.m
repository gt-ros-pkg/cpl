function c = cost_lateexpensive( d )
%COST_LATEEXPENSIVE Summary of this function goes here
%   Detailed explanation goes here
    
    if d < 0 % early
        c = 1 * abs(d) ^ 2;

    else % late
        c = 5 * abs(d) ^ 7;
    end
   

end

