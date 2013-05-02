





%% receive
if ~exist('executedplan')
    executedplan.events = [];
end
if planningconnt.BytesAvailable > 0
    len = fread(planningconnt, 1, 'int');
    executedplan = char(fread(planningconnt, len, 'char'))';
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


%% gen plan
plan1.t0 = t0;
plan1.events = struct( ...
            'signature', {111, 222, 333, 444, 98532, 324, 3214}, ...
            'name', {'Body', 'Nose_A', 'Nose_H', 'Body', 'Wing_H', 'Nose_A', 'Tail_H'}, ...
            'bin_id', {3, 11, 10, 3, 7, 11, 13}, ...
            'location', {'L11', 'L12', 'L13', 'L26', 'L11', 'L24', 'L12'}, ...
            'type', {'start_distribution', 'start_distribution', 'start_distribution', ...
                     'end_distribution', 'start_distribution', 'end_distribution', 'start_distribution'}, ...
            'cost_type', {'cost_lateexpensive', 'cost_lateexpensive', 'cost_lateexpensive', ...
                          'cost_earlyexpensive', 'cost_lateexpensive', 'cost_earlyexpensive', 'cost_lateexpensive'}, ...
            'pre_duration', num2cell(ones(1, 7) * 5 * 30 / 5), ...
            'post_duration', num2cell(ones(1, 7) * 1 * 30 / 5));

plan2.t0 = t0;
plan2.events = struct( ...
            'signature', {111, 222, 333, 444, 555, 666, 777, 888, 1}, ...
            'name', {'Body', 'Nose_A', 'Nose_H', 'Body', 'Nose_H', 'Wing_AT', 'Wing_AD', 'Nose_A', 'Tail_AT'}, ...
            'bin_id', {3, 11, 10, 3, 10, 12, 2, 11, 14}, ...
            'location', {'L11', 'L12', 'L13', 'L26', 'L24', 'L11', 'L13', 'L36', 'L12'}, ...
            'type', {'start_distribution', 'start_distribution', 'start_distribution', ...
                     'end_distribution', 'end_distribution', ...
                     'start_distribution', 'start_distribution', ...
                     'end_distribution', 'start_distribution'}, ...
            'cost_type', {'cost_lateexpensive', 'cost_lateexpensive', 'cost_lateexpensive', ...
                          'cost_earlyexpensive', 'cost_earlyexpensive', ...
                          'cost_lateexpensive', 'cost_lateexpensive', ...
                          'cost_earlyexpensive', 'cost_lateexpensive'}, ...
            'pre_duration', num2cell(ones(1, 9) * 5 * 30 / 5), ...
            'post_duration', num2cell(ones(1, 9) * 1 * 30 / 5));
        

plan3.t0 = t0;
plan3.events = struct( ...
            'signature', {111, 222, 333, 444, 555, 666, 777, 888, 2}, ...
            'name', {'Body', 'Nose_A', 'Nose_H', 'Body', 'Nose_H', 'Wing_AT', 'Wing_AD', 'Nose_A', 'Tail_AD'}, ...
            'bin_id', {3, 11, 10, 3, 10, 12, 2, 11, 15}, ...
            'location', {'L11', 'L12', 'L13', 'L26', 'L24', 'L11', 'L13', 'L36', 'L12'}, ...
            'type', {'start_distribution', 'start_distribution', 'start_distribution', ...
                     'end_distribution', 'end_distribution', ...
                     'start_distribution', 'start_distribution', ...
                     'end_distribution', 'start_distribution'}, ...
            'cost_type', {'cost_lateexpensive', 'cost_lateexpensive', 'cost_lateexpensive', ...
                          'cost_earlyexpensive', 'cost_earlyexpensive', ...
                          'cost_lateexpensive', 'cost_lateexpensive', ...
                          'cost_earlyexpensive', 'cost_lateexpensive'}, ...
            'pre_duration', num2cell(ones(1, 9) * 5 * 30 / 5), ...
            'post_duration', num2cell(ones(1, 9) * 1 * 30 / 5));


plans = [plan1 plan2 plan3];

%% 

clearvars bestplan
bestplan.t0 = t0;
bestplan.events = plans(1).events(1:0);
bestplan.score = -inf;

for i=1:length(plans)
    
    plans(i).valid = 1;
    
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
    plans = nx_assign_struct(plans, i, find_optimal_timing_for_order(plans(i), data));

    % calculate score
    if i == 1
        s = get_symbol_by_name(data.grammar, 'Tail_H');
        plans(i).score = sum(s.start_distribution);
    elseif i == 2
        s = get_symbol_by_name(data.grammar, 'Tail_AT');
        plans(i).score = sum(s.start_distribution);
    elseif i == 3
        s = get_symbol_by_name(data.grammar, 'Tail_AD');
        plans(i).score = sum(s.start_distribution);
    end

    % best?
    if bestplan.score < plans(i).score
        bestplan = plans(i);
    end
   
end


%% SEND
costx = bestplan;
bestplan.costx = [];
planning_s     = nx_toxmlstr(bestplan);
fwrite(planningconnt, length(planning_s), 'int'); 
fwrite(planningconnt, planning_s, 'char');








