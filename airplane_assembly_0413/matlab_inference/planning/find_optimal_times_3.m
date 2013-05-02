function [optimalcost_id mintotalcost] = find_optimal_times_3(costx, t0, d)
%FIND_OPTIMAL_TIMES Summary of this function goes here
%   Detailed explanation goes here



n  = size(costx, 1);
T  = size(costx, 2);


costx(1, 1:t0+d(1)) = inf;


[dummy optimalcost_id] = min(costx, [], 2);

% check for d constraint
consecutive_point = nan;
for i=2:n
    if optimalcost_id(i) - optimalcost_id(i-1) < d(i)
        consecutive_point = i;
        break;
    end
end

% recusive call
if ~isnan(consecutive_point)
    
    consecutive_point_newcost = costx(consecutive_point,:) + ...
        [inf*ones(1, d(consecutive_point)) costx(i-1,1:end-d(consecutive_point))];
    newcostx = costx;
    newcostx(consecutive_point,:) = consecutive_point_newcost;
    newcostx(consecutive_point-1,:) = [];
    newd = d;
    newd(consecutive_point) = newd(consecutive_point) + newd(consecutive_point-1);
    newd(consecutive_point-1) = [];
    
    newoptimalcost_id = find_optimal_times_3(newcostx, t0, newd);
    optimalcost_id(1:consecutive_point-2) = newoptimalcost_id(1:consecutive_point-2);
    optimalcost_id(consecutive_point:end) = newoptimalcost_id(consecutive_point-1:end);
    optimalcost_id(consecutive_point-1) = optimalcost_id(consecutive_point) - d(consecutive_point);
end

% ok dokie
mintotalcost = 0;
for i=1:n
    mintotalcost = mintotalcost + costx(i, optimalcost_id(i));
end

end

