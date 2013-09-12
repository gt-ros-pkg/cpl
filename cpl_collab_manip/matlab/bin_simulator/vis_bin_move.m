clear

hand_off = 0.1;
df_reach_off = homo2d(-0.1, 0.0, 0.0);
df_table_width = 0.8;
df_bin_width = 0.1;
row1_off = 0.5;
row2_off = 0.9;
row3_offx = 0.9;
row3_offy = 1.1;
row3_offr = -pi/4.0;

slot_template = struct('row', -1, 'center', df_reach_off, 'reach_loc', df_reach_off);

slots(1:3+4*2) = slot_template;

row_nums_cell = num2cell(1*ones(1,3));
[slots(1:3).row] = row_nums_cell{:};
row1_center = homo2d(row1_off, 0.0, 0.0);
row_centers_cell = get_row_slots(3, df_table_width, df_bin_width, row1_center);
[slots(1:3).center] = row_centers_cell{:};

row_nums_cell = num2cell(2*ones(1,4));
[slots(4:7).row] = row_nums_cell{:};
row2_center = homo2d(row2_off, 0.0, 0.0);
row_centers_cell = get_row_slots(4, df_table_width, df_bin_width, row2_center);
[slots(4:7).center] = row_centers_cell{:};

row_nums_cell = num2cell(3*ones(1,4));
[slots(8:11).row] = row_nums_cell{:};
row3_center = homo2d(row3_offx, row3_offy, row3_offr);
row_centers_cell = get_row_slots(4, df_table_width, df_bin_width, row3_center);
[slots(8:11).center] = row_centers_cell{:};

for i = 1:numel(slots)
    reach_loc = slots(i).center * df_reach_off;
    slots(i).reach_loc = reach_loc;
    reach_hum_loc = homo2d(hand_off, 0.0, 0.0)^-1 * reach_loc;
    slots(i).task_frame = homo2d(hand_off, 0.0, atan2(reach_hum_loc(2,3),reach_hum_loc(1,3)));
    slots(i).hand_dist = norm([reach_hum_loc(1:2,3)]);
end

all_reach_locs = cell2mat({slots(:).reach_loc});
figure(2)
clf
hold on
xlim([-1.5, 0.5])
ylim([-0.5, 1.5])
plot(-all_reach_locs(2,3:3:size(all_reach_locs,2))', all_reach_locs(1,3:3:size(all_reach_locs,2))', 'x')
plot(0.0, hand_off, 'rx')

p = [0.8*slots(3).hand_dist, 0.0, 1.0]';
p2 = slots(3).task_frame * p;
plot(-p2(2), p2(1), '+g')

vel_robot = 1.0;
dur_lift = 1.1;
dur_grasp = 0.8;

% bin_states records the state of the bin at the corresponding interval.
% The state index corresponds to the end of the interval expressed by 
% bin_times. Positive numbers are slot indexes, negatives represent when
% the bin is held by the robot, and nan is a placeholder for the beginning
% of the array.
bin_times{1} = [-inf, 2, 6, 8, inf];
bin_states{1} = [nan, 1, -1, 2, -1];
bin_times{2} = [-inf, 2, inf];
bin_states{2} = [nan, -1, 3];

% find bin availability from the bins which are in row 1
row1_slots = find([slots.row] == 1);
for i = 1:numel(bin_times)
    row1_state_inds = find(ismember(bin_states{i}, row1_slots));
    bin_avail{i} = zeros(1,numel(row1_state_inds)*2);
    bin_avail{i}(1:2:end-1) = bin_times{i}(row1_state_inds-1);
    bin_avail{i}(2:2:end) = bin_times{i}(row1_state_inds);
end

robplan_template = struct('bin_ind', nan, 'target_slot', nan, 'source_slot', nan, ...
                          'time_act', nan, 'has_wait', nan);
robplan = struct([]);
%robplan(1) = robplan_template;

roboacts_template = struct('time', nan, 'bin_ind', nan, 'target_slot', nan, ...
                           'source_slot', nan, 'prev_slot', nan);
robacts(1:2) = roboacts_template;
robacts(1).time = -inf;
robacts(1).target_slot = 1; % robot starts hovering above slot 1
robacts(2).time = inf;
robacts(2).bin_ind = -1;

% code to add new robot action
% input: bin_ind, target_slot, source_slot, time_act, has_wait

% previous slot is the last place robot delivered to
prev_slot = robacts(end-1).target_slot; 
if has_wait
    % add 2 states, end the last waiting time (old end) when the move starts
    robacts(end+1:end+2) = robacts(end);
    robacts(end-2).time = time_act;
else
    robacts(end+1) = robacts(end);
end
dist_grasp = norm(slots(prev_slot).center(1:2,3)-slots(source_slot).center(1:2,3));
dist_deliv = norm(slots(source_slot).center(1:2,3)-slots(target_slot).center(1:2,3));
time_end = time_act + dist_grasp/vel_robot + dist_deliv/vel_robot + 2*dur_grasp + 4*dur_lift;
rm_time = time_act + dist_grasp/vel_robot + dur_lift + dur_grasp;
dv_time = time_end - (dur_lift + dur_grasp);
robacts(end-1).time = time_end;
robacts(end-1).bin_ind = bin_ind;
robacts(end-1).target_slot = target_slot;
robacts(end-1).source_slot = source_slot;
robacts(end-1).prev_slot = prev_slot;
robacts(end-1).rm_time = rm_time; % remove time
robacts(end-1).dv_time = dv_time; % deliver time

bin_init_slots = [1, 4];

% binstates records the bins' movement history
binstates_template = struct('time', nan, 'slot', nan);
for i = 1:numel(bin_init_slots)
    binstates{i}(1:2) = binstates_template;
    binstates{i}(1).time = -inf;
    binstates{i}(2).time = inf;
    binstates{i}(2).slot = bin_init_slots(i);
end

% determine binstates from robacts robot action history
for i = 2:numel(robacts)-1
    if robacts(i).bin_ind < 0
        continue
    end
    bin_ind = robacts(i).bin_ind;
    binstates{bin_ind}(end+1:end+2) = binstates_template;
    binstates{bin_ind}(end-2).time = robacts(i).rm_time;
    binstates{bin_ind}(end-1).time = robacts(i).dv_time;
    binstates{bin_ind}(end-1).slot = -1;
    binstates{bin_ind}(end).time = inf;
    binstates{bin_ind}(end).slot = robacts(i).target_slot;
end

% find bin availability from the bins which are in row 1
row1_slots = find([slots.row] == 1);
for i = 1:numel(binstates)
    row1_state_inds = find(ismember([binstates{i}.slot], row1_slots));
    binavail{i} = zeros(1,numel(row1_state_inds)*2);
    times = [binstates{i}.time];
    binavail{i}(1:2:end-1) = times(row1_state_inds-1);
    binavail{i}(2:2:end) = times(row1_state_inds);
end
