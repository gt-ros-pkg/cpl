function c = cost_earlyexpensive( d )
%COST_EARLYEXPENSIVE Summary of this function goes here
%   Detailed explanation goes here

    if d < 0 % early
        c = 3 * abs(d) ^ 6;

    else % late
        c = 1 * abs(d) ^ 2;
    end
   
end

