function [optimalcost_id mintotalcost] = find_optimal_times_2(costx, t0, d)
%FIND_OPTIMAL_TIMES Summary of this function goes here
%   Detailed explanation goes here



n  = size(costx, 1);
T  = size(costx, 2);


costx(1, 1:t0+d(1)) = inf;

mintotalcost = +inf;
optimalcost_id = nan(1, n);


for i=1:500
    
    a = nan(1, n);
    c = 0;
    
    t = t0;
    
    % gen random
    for j=1:n
        t = t + d(j);
        %disp('---');
        %disp(j);
        %disp(t)
        %disp( T-sum(d(j+1:end)));
        a(j) = randi([t T-sum(d(j+1:end))]);
        c = c + costx(j, a(j));
        t = a(j);
    end
    
    % try to move to local minimum
    movenum = 0;
    while 1
        movenum = movenum + 1;
        j = randi([1 n]);
        if a(j)+1 < T && costx(j, a(j)+1) < costx(j, a(j))
            if j == n || a(j+1) - (a(j)+1) >= d(j+1)
                a(j) = a(j)+1;
                movenum = 0;
            end
        end
        if a(j)-1 > 1 && costx(j, a(j)-1) < costx(j, a(j))
            if j == 1 || (a(j)-1) - a(j-1) >= d(j)
                a(j) = a(j)-1;
                movenum = 0;
            end
        end
        if movenum > 30,
            break;
        end
    end
    
    % check
    c = 0;
    for j=1:n
        c = c + costx(j, a(j));
    end
    if c < mintotalcost,
        mintotalcost = c;
        optimalcost_id = a;
    end;
end

end

