function [robacts, binstates, binavail, availslot] = gen_rob_bin_states(robplan, bin_init_slots, slots)

% params
vel_robot = 1.0;
dur_lift = 1.1;
dur_grasp = 0.8;

row1_slots = find([slots.row] == 1);
row_rest_slots = find([slots.row] ~= 1);

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
for i = 1:numel(robplan.act_types)
    bin_ind = robplan.bin_inds(i);
    act_type = robplan.act_types(i);
    time_act = robplan.times(i);
    if act_type == 0
        has_wait = 1;
        continue
    elseif act_type == 1 % deliver
        empty_row1_slots = row1_slots(find(ismember(row1_slots, bin_cur_slots) == 0));
        target_slot = empty_row1_slots(1);
    else % remove
        empty_row_rest_slots = row_rest_slots(find(ismember(row_rest_slots, bin_cur_slots) == 0));
        target_slot = empty_row_rest_slots(1);
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
    slot_locations = [binstates{i}.slot]; % array of slot locations for each bin state
    % indices in the bin states which are located at row1
    row1_state_inds = find(ismember(slot_locations, row1_slots)); 
    binavail{i} = zeros(1,numel(row1_state_inds)*2);
    availslot{i} = -1*ones(1,numel(row1_state_inds)*2);
    times = [binstates{i}.time];
    binavail{i}(1:2:end-1) = times(row1_state_inds-1);
    binavail{i}(2:2:end) = times(row1_state_inds);
    availslot{i}(2:2:end) = slot_locations(row1_state_inds);
end
