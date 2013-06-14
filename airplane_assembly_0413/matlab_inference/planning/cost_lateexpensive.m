function c = cost_lateexpensive( d )
%COST_LATEEXPENSIVE Summary of this function goes here
%   Detailed explanation goes here

    if abs(d) > 50
        d = d / abs(d) * 50;
    end
    

    if d < 0 % early
        c = 1 * abs(d) ^ 1.1;
    else % late
        c = 10 * abs(d) ^ 3;
    end
   

end

