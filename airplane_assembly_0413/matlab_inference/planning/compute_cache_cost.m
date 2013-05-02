
clear; load d; clc

T = data.T;

planning.cache_cost.cost_squareddist    = zeros(2*T + 1, 1);
planning.cache_cost.cost_earlyexpensive = zeros(2*T + 1, 1);
planning.cache_cost.cost_lateexpensive  = zeros(2*T + 1, 1);
for i=1:2*T + 1
    planning.cache_cost.cost_squareddist(i)    = cost_squareddist(i-T-1);
    planning.cache_cost.cost_earlyexpensive(i) = cost_earlyexpensive(i-T-1);
    planning.cache_cost.cost_lateexpensive(i)  = cost_lateexpensive(i-T-1);
    planning.cache_cost.cost_zeros(i)          = cost_zeros(i-T-1);
end

data.planning = planning;


%%

orderx.t0 = 0;
orderx.events = struct( ...
    'name', {'Body', 'Nose_A', 'Nose_H', 'Body'}, ...
    'type', {'start_distribution', 'start_distribution', 'start_distribution', 'end_distribution'}, ...
    'cost_type', {'cost_lateexpensive', 'cost_lateexpensive', 'cost_lateexpensive', 'cost_earlyexpensive'}, ...
    'mindist', {0, 30, 30, 20});

%% process

hold on;

orderx.costx = zeros(length(orderx.events), data.T);
for i=1:length(orderx.events)
    sid = actionname2symbolid(orderx.events(i).name, data.grammar);
    d   = data.grammar.symbols(sid).(orderx.events(i).type);
    orderx.costx(i,:) = ef2(d, planning.cache_cost.(orderx.events(i).cost_type));
    
    plot(d);
end
[orderx.optimal_t orderx.mintotalcost] = find_optimal_times(orderx.costx, orderx.t0, [orderx.events.mindist]);


orderxx = find_optimal_timing_for_order(orderx, data);

plot(orderx.optimal_t, [0 0 0 0], 'g*');












