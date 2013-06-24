function [optimalcost_id mintotalcost] = find_optimal_times_4(costx, t0, d)
%FIND_OPTIMAL_TIMES Summary of this function goes here
%   Detailed explanation goes here


n  = size(costx, 1);
T  = size(costx, 2);

if n == 0
    optimalcost_id  = [];
    mintotalcost    = 0;
    return;
end;

assert(t0 < size(costx,2));

if t0 + d(1) > size(costx,2)
    d(1) = size(costx,2) - t0;
end

costx(1, 1:t0+d(1)) = inf;

if n == 1
    [mintotalcost optimalcost_id] = min(costx);
    if isinf(mintotalcost)
        optimalcost_id = size(costx,2);
    end
    return;
end


% combine the first two
cost12combine       = inf(1, T);
fromid2_to_best_id1 = nan(1, T);

for id1=t0+d(1)+1:T
    for id2=id1+d(2):T
        if costx(1,id1) + costx(2,id2) < cost12combine(id2)
            cost12combine(id2)          = costx(1,id1) + costx(2,id2);
            fromid2_to_best_id1(id2)    = id1;
        end
    end
end


% recursive call
new_costx = [cost12combine; costx(3:end,:)];
new_d     = [d(1)+d(2), d(3:end)];

[optimalcost_id mintotalcost] = find_optimal_times_4(new_costx, t0, new_d);

% get back the first two
if isnan(fromid2_to_best_id1(optimalcost_id(1)))
    optimalcost_id = [optimalcost_id(1) optimalcost_id];
else
    optimalcost_id = [fromid2_to_best_id1(optimalcost_id(1)) optimalcost_id];
end

end

