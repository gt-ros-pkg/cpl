function [optimalcost_id mintotalcost] = find_optimal_times(costx, t0, d)
%FIND_OPTIMAL_TIMES Summary of this function goes here
%   Detailed explanation goes here

[optimalcost_id  mintotalcost] = find_optimal_times_3(costx, t0, d);
return;

n  = size(costx, 1);
T  = size(costx, 2);

costx(1, 1:t0+d(1)) = inf;

optimalcost_id = nan(n, 1);


costn = nan(n, T);
conse = nan(n, 1);

for i=1:n
    
    if i == 1,
        costn(i,:) = costx(1,:);
        conse(i) = 0;
        continue;
    end
    
    [dummy mincosti] = min(costx(i,:));
    [dummy mincostp] = min(costn(i-1,:));
    
    if mincosti - mincostp > d(i)
        costn(i,:) = costx(i,:);
        conse(i) = 0;
    else
        costn(i,:) = costx(i,:) + [inf*ones(1, d(i)) costn(i-1,1:end-d(i))];
        conse(i) = 1;
    end
    
end

conse(n+1) = 0;

for i=n:-1:1
    
    if conse(i+1) == 0
        [dummy optimalcost_id(i)] = min(costn(i,:));
    else
        optimalcost_id(i) = optimalcost_id(i+1) - d(i+1);
    end
    
end


%% compute total cost

mintotalcost = 0;

for i=1:n
    mintotalcost = mintotalcost + costx(i, optimalcost_id(i));
end


end

