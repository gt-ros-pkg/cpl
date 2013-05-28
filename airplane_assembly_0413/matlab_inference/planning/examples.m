
close all;
addpath(genpath('./../'));

%% cache cost functions

T = 1000;

planning.cache_cost.cost_squareddist    = zeros(2*T + 1, 1);
planning.cache_cost.cost_earlyexpensive = zeros(2*T + 1, 1);
planning.cache_cost.cost_lateexpensive  = zeros(2*T + 1, 1);
for i=1:2*T + 1
    planning.cache_cost.cost_squareddist(i)    = cost_squareddist(i-T-1);
    planning.cache_cost.cost_earlyexpensive(i) = cost_earlyexpensive(i-T-1);
    planning.cache_cost.cost_lateexpensive(i)  = cost_lateexpensive(i-T-1);
    planning.cache_cost.cost_zeros(i)          = cost_zeros(i-T-1);
end


%% make up the distributions

colors = rand(1000, 3);


action1_start = nxmakegaussian(T, 100, 100);
action1_end   = nxmakegaussian(T, 300, 500);
action2_start = action1_end * 0.4;
action3_start = action1_end * 0.6;
action2_end   = nxmakegaussian(T, 400, 500) * 0.4;
action3_end   = nxmakegaussian(T, 600, 5000) * 0.6;

%% make up a sequence
distributions = [action1_start; action2_start; action3_start; action3_end; action2_end; action1_end];
texts = {'action1 start', 'action2 start',  'action3 start', 'action3 end', 'action2 end','action1 end'};

% integrate cost function of delivery points
d_costs(1,:) = ef2(distributions(1,:), planning.cache_cost.cost_lateexpensive);
d_costs(2,:) = ef2(distributions(2,:), planning.cache_cost.cost_lateexpensive);
d_costs(3,:) = ef2(distributions(3,:), planning.cache_cost.cost_lateexpensive);
d_costs(4,:) = ef2(distributions(4,:), planning.cache_cost.cost_earlyexpensive);
d_costs(5,:) = ef2(distributions(5,:), planning.cache_cost.cost_earlyexpensive);
d_costs(6,:) = ef2(distributions(6,:), planning.cache_cost.cost_earlyexpensive);



%% joint optimization
distance_constraints = [50, 60, 30, 100, 30, 80]; % distance constraints between points
[optimal_t mintotalcost] = find_optimal_times(d_costs, 10, distance_constraints);

%% drawing

% draw distributions with naive optimal points
figure(1);
hold on;
for i=1:size(distributions, 1)
    plot(distributions(i,:), 'color', colors(i,:));
end
for i=1:size(distributions, 1)
    [~, id] = min(d_costs(i,:));
    plot(id, 0, '*', 'color', colors(i,:));
end
hold off;
legend(texts);

% draw integrated costs with naive optimal points
figure(2);
for i=1:size(distributions, 1)
    subplot(3,2,i);
    plot(d_costs(i,:), 'color', colors(i,:));
    [~, id] = min(d_costs(i,:));
    hold on; plot(id, 0, '*r'); hold off;
    legend({texts{i}, 'best point with min cost'});
end


% draw distributions with joint optimal points
figure(3);
hold on;
for i=1:size(distributions, 1)
    plot(distributions(i,:), 'color', colors(i,:));
end
for i=1:size(distributions, 1)
    plot(optimal_t(i), 0, '*', 'color', colors(i,:));
end
hold off;
legend(texts);
























