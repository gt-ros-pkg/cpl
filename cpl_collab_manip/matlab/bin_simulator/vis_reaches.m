clear 
start_time = 5.0;
durs_step = [10.0, 3.0, 4.0, 5.0, 3.0, 5.0, 6.0];
step_bin =  [   1,   1,   1,   1,   1,   2,   2];
dists_bin = [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
binavail{1} = [-inf, 17, 22, inf];
binavail{2} = [-inf, 17, 40, inf];

df_reach_speed = 0.8;
df_draw_dur = 0.2;
df_retreat_speed = 0.8;

for i = 1:numel(durs_step)
    durs_draw(i) = df_draw_dur;
    vels_reach(i) = df_reach_speed;
    vels_retreat(i) = df_retreat_speed;
    dists_reach(i) = dists_bin(i);
end

no_time_between = zeros(size(durs_step));
% check to see if we need to accelerate reaches close together
for i = 2:numel(durs_step)
    dur_mid = dists_reach(i)/vels_reach(i) + durs_draw(i);
    last_retreat_dur = dists_bin(i-1)/vels_retreat(i-1);
    if dur_mid + last_retreat_dur > durs_step(i-1)
        vels_retreat(i-1) = 2.0*dists_reach(i-1)/(durs_step(i-1) - durs_draw(i));
        vels_reach(i) = 2.0*dists_reach(i)/(durs_step(i-1) - durs_draw(i));
        no_time_between(i) = 1;
    end
end

reach_final_mid_times = zeros(1, numel(durs_step));
% reach_mid = durs_step(1);
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
    if i_rch > numel(durs_step)
        % finished, exit
        if reach_final_mid_times(i_rch-1) + durs_step(i_rch-1) > humacts(end).time
            % add the last assembly time
            new_action = action_template;
            new_action.time = reach_final_mid_times(i_rch-1) + durs_step(i_rch-1);
            new_action.type = 2;
            humacts(end+1) = new_action;
        end
        break
    end

    bin_ind = step_bin(i_rch);

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
    if i_rch == 1
        reach_mid = start_time;
    else
        reach_mid = reach_final_mid_times(i_rch-1) + durs_step(i_rch-1);
    end
    reach_start = reach_mid - (dists_reach(i_rch)/vels_reach(i_rch)+durs_draw(i_rch));
    reach_end = reach_mid + dists_reach(i_rch)/vels_retreat(i_rch);
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
        new_action.dist_reach = dists_reach(i_rch);
        new_action.vel_reach = vels_reach(i_rch);
        new_action.dur_draw = durs_draw(i_rch);
        new_action.vel_retreat = vels_retreat(i_rch);
        humacts(end+1) = new_action;

        reach_final_mid_times(i_rch) = action_start + reach_mid - reach_start;
        assem_after_reach = 1;
        i_rch = i_rch + 1;
    else
        % fail from action_start to avail_end
        new_dist_reach = min(dists_reach(i_rch), ...
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

figure(1)
clf
for i = 1:numel(binavail)
    subplot(numel(binavail),1,i)
    hold on
    xlim([0 70])
    ylim([0 1.4])
end

for i = 1:numel(humacts)
    for j = 1:numel(binavail)
        subplot(numel(binavail),1,j)
        if humacts(i).type == 0 
            plot(humacts(i).time*[1 1], [0 1.3], 'm')
        elseif humacts(i).type == 3 
            plot(humacts(i).time*[1 1], [1.1 1.5], 'b')
            plot([humacts(i-1).time humacts(i).time], [1.3 1.3], 'b', 'LineWidth', 8)
            plot([humacts(i-1).time humacts(i).time], [0.05 0.05], 'b')
        elseif humacts(i).type == 2
            plot(humacts(i).time*[1 1], [1.1 1.5], 'k')
            plot([humacts(i-1).time humacts(i).time], [1.3 1.3], 'k', 'LineWidth', 8)
            t = linspace(humacts(i-1).time, humacts(i).time, 100);
            plot(t, abs(0.1*rand(size(t))), 'k')
        elseif (humacts(i).type == 1 || humacts(i).type == 4)
            if humacts(i).bin_ind == j
                if humacts(i).type == 1
                    color = 'g';
                else
                    color = 'r';
                end
                dur_reach = humacts(i).dist_reach/humacts(i).vel_reach;
                dur_mid = dur_reach+humacts(i).dur_draw;
                dur_retreat = humacts(i).dist_reach/humacts(i).vel_retreat;
                dur_total = dur_mid+dur_retreat;
                action_start_time = humacts(i).time - dur_total;
                t1 = linspace(0, dur_reach, 100);
                plot(t1+action_start_time, humacts(i).vel_reach*t1,color)
                t2 = linspace(dur_reach, dur_mid, 100);
                plot(t2+action_start_time, humacts(i).dist_reach+t2*0,color)
                t3 = linspace(dur_mid, dur_total, 100);
                plot(t3+action_start_time, humacts(i).dist_reach-humacts(i).vel_retreat*(t3-dur_mid),color)
                if humacts(i).type == 1
                    plot(humacts(i).time*[1 1], [1.1 1.5], 'g')
                    plot([humacts(i-1).time humacts(i).time], [1.3 1.3], 'g', 'LineWidth', 8)
                else
                    plot(humacts(i).time*[1 1], [1.1 1.5], 'r')
                    plot([humacts(i-1).time humacts(i).time], [1.3 1.3], 'r', 'LineWidth', 8)
                end
            end
        end
    end
end
