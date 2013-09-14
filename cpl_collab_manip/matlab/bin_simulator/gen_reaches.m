function [humacts] = gen_reaches(humplan, binavail, availslot, slots)

% parameters
df_reach_speed = 0.8;
df_draw_dur = 0.2;
df_retreat_speed = 0.8;

for i = 1:numel(humplan.durs_step)
    durs_draw(i) = df_draw_dur;
    vels_reach(i) = df_reach_speed;
    vels_retreat(i) = df_retreat_speed;
end

% check to see if we need to accelerate reaches close together
% first find the length of the longest reach to find an upper bound for distances
row1_slots = find([slots.row] == 1);
max_row1_dist = max([slots(row1_slots).hand_dist]);

no_time_between = zeros(size(humplan.durs_step));
for i = 2:numel(humplan.durs_step)
    dur_mid = max_row1_dist/vels_reach(i) + durs_draw(i);
    last_retreat_dur = max_row1_dist/vels_retreat(i-1);
    if dur_mid + last_retreat_dur > humplan.durs_step(i-1)
        vels_retreat(i-1) = 2.0*max_row1_dist/(humplan.durs_step(i-1) - durs_draw(i));
        vels_reach(i) = 2.0*max_row1_dist/(humplan.durs_step(i-1) - durs_draw(i));
        no_time_between(i) = 1;
    end
end

reach_final_mid_times = zeros(1, numel(humplan.durs_step));
% reach_mid = humplan.durs_step(1);
i_avail = num2cell(2*ones(1,numel(binavail)));
i_rch = 1;
wait_after_fail = 0; % we're waiting from the last iteration
assem_after_reach = 0; % we're assembling from the last iteration

action_template = struct('time', -1.0, 'type', -1, 'bin_ind', -1, 'dist_reach', 0.0, ...
                         'vel_reach', 0.0, 'dur_draw', -1.0, 'vel_retreat', 0.0);
% humacts is the structure array of human actions the human will perform
humacts(1) = action_template;
humacts(1).type = 0;
% action type: nothing = 0, reach = 1, assem = 2, wait = 3, fail = 4

while 1
    if i_rch > numel(humplan.durs_step)
        % finished, exit
        if reach_final_mid_times(i_rch-1) + humplan.durs_step(i_rch-1) > humacts(end).time
            % add the last assembly time
            new_action = action_template;
            new_action.time = reach_final_mid_times(i_rch-1) + humplan.durs_step(i_rch-1);
            new_action.type = 2;
            humacts(end+1) = new_action;
            last_action = action_template;
            last_action.type = 0;
            last_action.time = inf;
            humacts(end+1) = last_action;
        end
        break
    end

    bin_ind = humplan.step_bin(i_rch);

    if i_avail{bin_ind} > numel(binavail{bin_ind})
        % no longer available, wait forever
        new_action = action_template;
        new_action.time = inf;
        new_action.type = 3;
        new_action.bin_ind = bin_ind;
        humacts(end+1) = new_action;
        break
    end

    avail_start = binavail{bin_ind}(i_avail{bin_ind}-1);
    avail_end = binavail{bin_ind}(i_avail{bin_ind});
    avail_slot = availslot{bin_ind}(i_avail{bin_ind});
    dist_reach = slots(avail_slot).hand_dist;
    if i_rch == 1
        reach_mid = humplan.start_time;
    else
        reach_mid = reach_final_mid_times(i_rch-1) + humplan.durs_step(i_rch-1);
    end
    reach_start = reach_mid - (dist_reach/vels_reach(i_rch)+durs_draw(i_rch));
    reach_end = reach_mid + dist_reach/vels_retreat(i_rch);
    if i_rch == 1
        humacts(1).time = reach_start;
    end
    action_start = max(avail_start, reach_start);
    wait_after_success = reach_start < avail_start;
    if reach_start > avail_end
        % wrong interval, move to next
        i_avail{bin_ind} = i_avail{bin_ind} + 2;
        continue
    end
    if assem_after_reach && ~no_time_between(i_rch)
        % only assemble after part reach where there's time inbetween reaches
        new_action = action_template;
        new_action.time = reach_start;
        new_action.type = 2;
        humacts(end+1) = new_action;
        assem_after_reach = 0;
    end
    if wait_after_success || wait_after_fail
        % wait from reach_start to avail_start
        new_action = action_template;
        new_action.time = avail_start;
        new_action.type = 3;
        new_action.bin_ind = bin_ind;
        humacts(end+1) = new_action;
        wait_after_fail = 0;
    end
    if reach_mid - reach_start <= avail_end - action_start
        % reach from action_start to action_start + reach_mid - reach_start
        new_action = action_template;
        new_action.time = action_start + reach_end - reach_start;
        new_action.type = 1;
        new_action.bin_ind = bin_ind;
        new_action.dist_reach = dist_reach;
        new_action.vel_reach = vels_reach(i_rch);
        new_action.dur_draw = durs_draw(i_rch);
        new_action.vel_retreat = vels_retreat(i_rch);
        humacts(end+1) = new_action;

        reach_final_mid_times(i_rch) = action_start + reach_mid - reach_start;
        assem_after_reach = 1;
        i_rch = i_rch + 1;
    else
        % fail from action_start to avail_end
        new_dist_reach = min(dist_reach, ...
                             vels_reach(i_rch)*(avail_end-action_start));
        new_action = action_template;
        new_action.time = avail_end+new_dist_reach/vels_retreat(i_rch);
        new_action.type = 4;
        new_action.bin_ind = bin_ind;
        new_action.dist_reach = new_dist_reach;
        new_action.vel_reach = vels_reach(i_rch);
        new_action.dur_draw = 0.0;
        new_action.vel_retreat = vels_retreat(i_rch);
        humacts(end+1) = new_action;

        wait_after_fail = 1;
        i_avail{bin_ind} = i_avail{bin_ind} + 2;
    end
end
