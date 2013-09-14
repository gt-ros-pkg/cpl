
% clc;
% load d2;

MAX_BIN_AVAILABLE_NUM = 3;

ACTION_PRE_DURATION  = round(9 * 30 / m.params.downsample_ratio);
ACTION_POST_DURATION = round(4 * 30 / m.params.downsample_ratio);

% compute bin distribution
n.bin_distributions = extract_bin_requirement_distributions( m );
N = length(n.bin_distributions);

% compute bin available
for i=1:N
    n.bin_distributions(i).bin_available = 0;
    
    b = n.bin_distributions(i).bin_id;
    if ~isempty(frame_info.bins(b).H)
    	d = norm([-1, -1.3] - [frame_info.bins(b).pq(1), frame_info.bins(b).pq(2)]);
    	condition_no = d > 0.9;
        if d < 0.9,
            n.bin_distributions(i).bin_available = 1;
        end
    end
    
    
    % use executed action
    if 1 & isfield(n, 'executedplan')
        for j=length(n.executedplan.events):-1:max(1, length(n.executedplan.events))
            if n.executedplan.events(j).bin_id == b,
                if n.executedplan.events(j).sname(1) == 'A'
                	n.bin_distributions(i).bin_available = 1;
                else
                	n.bin_distributions(i).bin_available = 0;   
                end
            end
        end
    end
end

% cut the past
for i=1:N
    
    n.bin_distributions(i).prob_not_finished = sum(n.bin_distributions(i).bin_nolonger_needed(nt+1:end));
        
    if ~n.bin_distributions(i).bin_available
        n.bin_distributions(i).bin_nolonger_needed(1:nt) = 0;
        n.bin_distributions(i).bin_needed = n.bin_distributions(i).bin_needed / sum(n.bin_distributions(i).bin_needed) * n.bin_distributions(i).prob_not_finished;
        
    end
end


% calculate expected start & end
for i=1:N
    d = n.bin_distributions(i).bin_needed;
    d = d / sum(d);
    n.bin_distributions(i).expected_start = ceil(sum(d .* [1:length(d)]));
    
    d = n.bin_distributions(i).bin_nolonger_needed;
    d = d / sum(d);
    n.bin_distributions(i).expected_end = ceil(sum(d .* [1:length(d)]));
end


%% special case: early-remove (ER94)
for i=1:N
    n.bin_distributions(i).ER94 = 0;
end

earliest_bin_id        = nan;
earliest_bin_start     = inf;
earliest_wsbin_id      = nan;
earliest_wsbin_start   = inf;
latest_wsbin_id        = nan;
latest_wsbin_start     = -inf;
bin_available_num      = 0;
ws_probs               = [];
ws_starts              = [];

for i=1:N
    
    if n.bin_distributions(i).bin_available
        bin_available_num = bin_available_num + 1;
        ws_probs(end+1)   = n.bin_distributions(i).prob_not_finished;
        ws_starts(end+1)  = n.bin_distributions(i).expected_start;
    end
    
    if n.bin_distributions(i).prob_not_finished > 0.5 & ...
            ~n.bin_distributions(i).bin_available & ...
            n.bin_distributions(i).expected_start < earliest_bin_start
        
        earliest_bin_id    = i;
        earliest_bin_start = n.bin_distributions(i).expected_start;
  
    end
    
    if n.bin_distributions(i).bin_available & ...
            n.bin_distributions(i).expected_start < earliest_wsbin_start
        
        earliest_wsbin_id    = i;
        earliest_wsbin_start = n.bin_distributions(i).expected_start;
  
    end
    
    if n.bin_distributions(i).bin_available & n.bin_distributions(i).expected_start > latest_wsbin_start 
        latest_wsbin_id    = i;
        latest_wsbin_start = n.bin_distributions(i).expected_start;
    end
end

if  ~isnan(earliest_bin_id) & ...
    sum(ws_probs >= n.bin_distributions(earliest_bin_id).prob_not_finished) >= MAX_BIN_AVAILABLE_NUM & ...
        sum(ws_starts >= n.bin_distributions(earliest_bin_id).expected_start) >= MAX_BIN_AVAILABLE_NUM
    
    warning(['ER94, remove ' num2str(latest_wsbin_id) ' for ' num2str(earliest_bin_id)]);
	
    n.bin_distributions(latest_wsbin_id).prob_not_finished      = 1;
    n.bin_distributions(latest_wsbin_id).bin_needed(:)          = 0;
    n.bin_distributions(latest_wsbin_id).bin_needed(1)          = 1;
    n.bin_distributions(latest_wsbin_id).bin_nolonger_needed(:) = 0;
    n.bin_distributions(latest_wsbin_id).bin_nolonger_needed(1) = 1;
    
end


%% cost integration
% nx_figure(1312);
for i=1:N
    n.bin_distributions(i).bin_needed_cost = ef2(n.bin_distributions(i).bin_needed, n.cache_cost.cost_lateexpensive);
    n.bin_distributions(i).bin_nolonger_needed_cost = ef2(n.bin_distributions(i).bin_nolonger_needed, n.cache_cost.cost_earlyexpensive);
    
%     subplot(N, 2, 2*i-1);
%     plot(n.bin_distributions(i).bin_needed);
%     hold on;
%     plot(n.bin_distributions(i).bin_nolonger_needed, 'r');
%     hold off;
%     subplot(N, 2, 2*i);
%     semilogy(n.bin_distributions(i).bin_needed_cost);
%     hold on;
%     semilogy(n.bin_distributions(i).bin_nolonger_needed_cost, 'r');
%     hold off;
end

%% gen plan
plans = {};
for i=1:100
    
    plan = gen_random_plan( n.bin_distributions );
    
    % check dup plan
    dup_plan = 0;
    for p=plans
        p = p{1};
        
        if length(p.actions) == length(plan.actions) & strcmp([p.actions.action_str], [plan.actions.action_str])
        
            if sum([p.actions.bin_id] == [plan.actions.bin_id]) == length(p.actions)
                dup_plan = 1;
                break;
            end
        end
    end
    if dup_plan
        continue;
    end
    
    % add plan
    plans{end+1} = plan;
    
    % disp
    if 0
        disp ------------new_plan---
        [n.bin_distributions.bin_available]
        for a=plan.actions
            %disp([a.action_str '  ' num2str(a.bin_id)]);
            disp([a.action_str '  ' binid2name( n.bin_distributions(a.bin_id).bin_id )]);
        end
    end
end


%% optimize plan

% add duration
for i=1:length(plans)
    
    costs     = zeros(length(plans{i}.actions), m.params.T);
    distances = zeros(length(plans{i}.actions));
    
   for j=1:length(plans{i}.actions)
       if plans{i}.actions(j).action_str(1) == 'A'
           plans{i}.actions(j).pre_duration  = ACTION_PRE_DURATION;
           plans{i}.actions(j).post_duration = ACTION_POST_DURATION;
       else
           plans{i}.actions(j).pre_duration  = ACTION_POST_DURATION;
           plans{i}.actions(j).post_duration = ACTION_PRE_DURATION;
       end
   end
    
end


for i=1:length(plans)
    
    % calculate cost and distance
    distances = zeros(1, length(plans{i}.actions));
    for j=1:length(plans{i}.actions)
       distances(j) = plans{i}.actions(j).pre_duration;
    end
    for j=1:length(plans{i}.actions)-1
       distances(j+1) = distances(j+1) + plans{i}.actions(j).post_duration;
    end
    
    % calculate cost
    costs = zeros(length(plans{i}.actions), m.params.T);
    for j=1:length(plans{i}.actions)
       if strcmp(plans{i}.actions(j).action_str, 'Add')
            costs(j,:) = n.bin_distributions(plans{i}.actions(j).bin_id).bin_needed_cost;
       else
            costs(j,:) = n.bin_distributions(plans{i}.actions(j).bin_id).bin_nolonger_needed_cost;
       end
    end
    
    % optimize
    [plans{i}.optimal_t plans{i}.local_cost] = find_optimal_times(costs, nt, distances);
    for j=1:length(plans{i}.actions)
       plans{i}.actions(j).optimal_t = plans{i}.optimal_t(j);
    end
    
    % find global cost
    plans{i}.global_cost  = plans{i}.local_cost;
    plans{i}.global_cost2 = 0;
    for b=1:N
        for action_str={'Add', 'Remove'}
            
            action_str = action_str{1};
            
            cost = zeros(1, m.params.T);
            
            if strcmp(action_str, 'Add')
                cost = n.bin_distributions(b).bin_needed_cost;
            else
                cost = n.bin_distributions(b).bin_nolonger_needed_cost;
            end
            
            if n.bin_distributions(b).bin_available & strcmp(action_str, 'Add')
                continue;
            end
            
            optimal_t = m.params.T;
            for action=plans{i}.actions
                if strcmp(action.action_str, action_str) & action.bin_id == b
                    % disp([action_str ' ' num2str(b)]);
                    optimal_t = action.optimal_t;
                end
            end
            
            plans{i}.global_cost2 = plans{i}.global_cost2 + cost(optimal_t);
            
            if optimal_t == m.params.T
                plans{i}.global_cost  = plans{i}.global_cost + cost(optimal_t);
            end
        end
    end
end


%% reorder by global cost
for i1=1:length(plans)
    for i2=i1+1:length(plans)
    	if plans{i1}.global_cost > plans{i2}.global_cost
            t = plans{i1};
            plans{i1} = plans{i2};
            plans{i2} = t;
        end
    end
end

% save
n.plans = plans;


%% convert best plan to old format
bestplan = struct;
for i=1:length(n.plans{1}.actions)
    bestplan.events(i).pre_duration  = n.plans{1}.actions(i).pre_duration;
    bestplan.events(i).post_duration = n.plans{1}.actions(i).post_duration;
    bestplan.events(i).sname         = [n.plans{1}.actions(i).action_str ' ' binid2name( n.bin_distributions(n.plans{1}.actions(i).bin_id).bin_id )];
    bestplan.events(i).bin_id        = n.bin_distributions(n.plans{1}.actions(i).bin_id).bin_id;
    bestplan.events(i).optimal_t     = n.plans{1}.actions(i).optimal_t;
end














