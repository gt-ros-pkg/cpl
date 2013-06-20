function orderx = find_optimal_timing_for_order(orderx, data, planning)
%FIND_OPTIMAL_TIMING_FOR_ORDER Summary of this function goes here
%   Detailed explanation goes here

n = length(orderx.events);
if n == 0
    return;
end

% prepare
orderx.costx = zeros(n, data.params.T);
for i=1:n
    sid = actionname2symbolid(orderx.events(i).name, data.grammar);
    
    if isfield(orderx.events(i), 'distribution')
        d = orderx.events(i).distribution;
    else
        d   = data.grammar.symbols(sid).(orderx.events(i).type);
    end
    
    
    orderx.costx(i,:) = ef2(d, planning.cache_cost.(orderx.events(i).cost_type));
end

mindist = [orderx.events.pre_duration 0] + [0 orderx.events.post_duration];
mindist = mindist(1:n);

% optimize
[optimal_t orderx.mintotalcost] = find_optimal_times(orderx.costx, orderx.t0, mindist);

% save
for i=1:n
    orderx.events(i).optimal_t = optimal_t(i);
end

end
