function n = n_planning_process( n, m, nt, frame_info )
%N_PLANNING_PROCESS Summary of this function goes here
%   Detailed explanation goes here


% return;

%% receive executed plan
if ~exist('executedplan')
    executedplan.events = [];
end
if n.ros_tcp_connection.BytesAvailable > 0
    len = fread(n.ros_tcp_connection, 1, 'int');
    executedplan = char(fread(n.ros_tcp_connection, len, 'char'))';
    executedplan = nx_fromxmlstr(executedplan);
end

%% compute t0

t0 = ceil(nt);

if length(executedplan.events) > 0
    
    t0 = max(t0, ...
             executedplan.events(end).matlab_execute_time + ...
                  executedplan.events(end).pre_duration + ...
                  executedplan.events(end).post_duration);
end



%% optimize plan

plans           = n.init.plans;
bestplan        = struct;
bestplan.t0     = t0;
bestplan.events = plans(1).events(1:0);
bestplan.score  = -inf;

for i=1:length(plans)
    
    plans(i).t0     = t0;
    plans(i).valid  = 1;
    
    % check valid
    if length(executedplan.events) > length(plans(i).events)
        plans(i).valid = 0;
        continue;
    end
    for j=1:length(executedplan.events)
        if plans(i).events(j).signature ~= executedplan.events(j).signature
            plans(i).valid = 0;
        end
    end
    
    if ~plans(i).valid
        continue;
    end
    
    % truncate
    plans(i).events(1:length(executedplan.events)) = [];

    % find optimal
    plans = nx_assign_struct(plans, i, find_optimal_timing_for_order(plans(i), m, n));

    % calculate score
    if i == 1
        s = get_symbol_by_name(m.grammar, 'Tail_H');
        plans(i).score = sum(s.start_distribution);
    elseif i == 2
        s = get_symbol_by_name(m.grammar, 'Tail_AT');
        plans(i).score = sum(s.start_distribution);
    elseif i == 3
        s = get_symbol_by_name(m.grammar, 'Tail_AD');
        plans(i).score = sum(s.start_distribution);
    end

    % best?
    if bestplan.score < plans(i).score
        bestplan = plans(i);
    end
   
end



%% send plan
n.executedplan = executedplan;
n.bestplan     = bestplan;
bestplan.costx = [];
planning_s     = nx_toxmlstr(bestplan);
fwrite(n.ros_tcp_connection, length(planning_s), 'int'); 
fwrite(n.ros_tcp_connection, planning_s, 'char');

end

