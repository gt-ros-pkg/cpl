slots = gen_slots();

% params
vel_robot = 1.0;
dur_lift = 1.1;
dur_grasp = 0.8;

bin_init_slots = [1, 4, 5];
% d 4, w, r 1, r 4
plan_bin_inds =  [   0,    2,    0,    1,   0,    2,    0 ];
plan_act_types = [   0,    1,    0,    2,   0,    2,    0 ];
plan_times =     [ 0.0, 10.0,  0.0, 25.0, 0.0, 40.0,  0.0 ];

row1_slots = find([slots.row] == 1);
row_rest_slots = find([slots.row] ~= 1);

% act_type: 0 = wait, 1 = deliver, 2 = remove
robplan_template = struct('bin_ind', nan, 'act_type', nan, 'time', nan);
% robplan = struct([]);
robplan(1:4) = robplan_template;
cell_bin_inds =  mat2cell(plan_bin_inds', ones(1,numel(plan_bin_inds)));
cell_act_types = mat2cell(plan_act_types', ones(1,numel(plan_act_types)));
cell_times =     mat2cell(plan_times', ones(1,numel(plan_times)));
[robplan(:).bin_ind] = cell_bin_inds{:};
[robplan(:).act_type] = cell_act_types{:};
[robplan(:).time] = cell_times{:};

roboacts_template = struct('bin_ind', nan, 'target_slot', nan, ...
                           'source_slot', nan, 'prev_slot', nan, ...
                           'start_time', nan, 'rm_time', nan, ...
                           'dv_time', nan, 'end_time', nan);
robacts(1:2) = roboacts_template;
robacts(1).end_time = -inf;
robacts(1).target_slot = 1; % robot starts hovering above slot 1
robacts(2).end_time = inf;
robacts(2).bin_ind = -1; % -1 = waits

% binstates records the bins' movement history
binstates_template = struct('time', nan, 'slot', nan);
for i = 1:numel(bin_init_slots)
    binstates{i}(1:2) = binstates_template;
    binstates{i}(1).time = -inf;
    binstates{i}(2).time = inf;
    binstates{i}(2).slot = bin_init_slots(i);
end

has_wait = 0;
bin_cur_slots = bin_init_slots;
for i = 1:numel(robplan)
    bin_ind = robplan(i).bin_ind
    act_type = robplan(i).act_type
    time_act = robplan(i).time
    if act_type == 0
        has_wait = 1;
        continue
    elseif act_type == 1 % deliver
        empty_row1_slots = row1_slots(find(ismember(row1_slots, bin_cur_slots) == 0))
        target_slot = empty_row1_slots(1)
    else % remove
        empty_row_rest_slots = row_rest_slots(find(ismember(row_rest_slots, bin_cur_slots) == 0))
        target_slot = empty_row_rest_slots(1)
    end
    source_slot = bin_cur_slots(bin_ind);

    % code to add new robot action
    % input: bin_ind, target_slot, source_slot, time_act, has_wait

    % previous slot is the last place robot delivered to
    prev_slot = robacts(end-1).target_slot; 
    if has_wait
        % add 2 states, end the last waiting time (old end) when the move starts
        robacts(end+1:end+2) = robacts(end);
        robacts(end-2).end_time = time_act;
        has_wait = 0;
    else
        robacts(end+1) = robacts(end);
    end
    dist_grasp = norm(slots(prev_slot).center(1:2,3)-slots(source_slot).center(1:2,3));
    dist_deliv = norm(slots(source_slot).center(1:2,3)-slots(target_slot).center(1:2,3));
    time_end = time_act + dist_grasp/vel_robot + dist_deliv/vel_robot + 2*dur_grasp + 4*dur_lift;
    rm_time = time_act + dist_grasp/vel_robot + dur_lift + dur_grasp;
    dv_time = time_end - (dur_lift + dur_grasp);
    robacts(end-1).start_time = time_act;
    robacts(end-1).rm_time = rm_time;
    robacts(end-1).dv_time = dv_time;
    robacts(end-1).end_time = time_end;
    robacts(end-1).bin_ind = bin_ind;
    robacts(end-1).target_slot = target_slot;
    robacts(end-1).source_slot = source_slot;
    robacts(end-1).prev_slot = prev_slot;

    % update binstates history
    binstates{bin_ind}(end+1:end+2) = binstates_template;
    binstates{bin_ind}(end-2).time = rm_time;
    binstates{bin_ind}(end-1).time = dv_time;
    binstates{bin_ind}(end-1).slot = -1;
    binstates{bin_ind}(end).time = inf;
    binstates{bin_ind}(end).slot = target_slot;
    bin_cur_slots(bin_ind) = target_slot;

end

% find bin availability from the bins which are in row 1
for i = 1:numel(binstates)
    row1_state_inds = find(ismember([binstates{i}.slot], row1_slots));
    binavail{i} = zeros(1,numel(row1_state_inds)*2);
    times = [binstates{i}.time];
    binavail{i}(1:2:end-1) = times(row1_state_inds-1);
    binavail{i}(2:2:end) = times(row1_state_inds);
end

figure(6)
subplot(3,1,1)
slots_hist_viz(binstates, slots, 0)
subplot(3,1,2)
rob_act_viz(robacts, slots, 0)
subplot(3,1,3)
bin_avail_viz(binavail, 0)
