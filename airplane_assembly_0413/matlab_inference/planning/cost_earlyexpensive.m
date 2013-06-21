function c = cost_earlyexpensive( d )
%COST_EARLYEXPENSIVE Summary of this function goes here
%   Detailed explanation goes here

    if abs(d) > 50
        d = d / abs(d) * 50;
    end
    
    if d < 0 % early
        c = 10 * abs(d) ^ 3;
    else % late
        c = 1 * abs(d) ^ 1;
    end
   
end

