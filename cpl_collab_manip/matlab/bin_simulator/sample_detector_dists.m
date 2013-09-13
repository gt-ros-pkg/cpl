function [detect_dists] = sample_detector_dists(humacts, slots, binavail, availslot, samp_interval, ...
                                              samp_num, detector_off)
samp_start = samp_interval(1);
samp_end = samp_interval(3);
samp_per_sec = (samp_num-1)/(samp_end-samp_start);
detector_slots = find([slots.row] == 1);
detect_dists = nan*ones(numel(binavail),samp_num);
slot_dists = nan*ones(numel(detector_slots),samp_num);

last_end_ind = 1;
for i = 2:numel(humacts)
    for d_slot_ind = 1:numel(detector_slots)
        d_slot = detector_slots(d_slot_ind);
        hand_dist = slots(d_slot).hand_dist;
        act_start = humacts(i-1).time;
        act_end = humacts(i).time;
        if act_end < samp_start
            continue
        end
        if act_start >= samp_end
            break
        end
        cur_start = max(samp_start, act_start);
        cur_end = min(samp_end, act_end);
        cur_start_ind = ceil((cur_start - samp_start) * samp_per_sec)+1;
        cur_end_ind = floor((cur_end - samp_start) * samp_per_sec)+1;
        cur_num = cur_end_ind-cur_start_ind+1;
        cur_type = humacts(i).type;

        if humacts(i).type == 0  % nothing
            slot_dists(d_slot,cur_start_ind:cur_end_ind) = hand_dist;
        elseif humacts(i).type == 3  % wait
            slot_dists(d_slot,cur_start_ind:cur_end_ind) = hand_dist;
        elseif humacts(i).type == 2 % assembly
            slot_dists(d_slot,cur_start_ind:cur_end_ind) = hand_dist + 0.04*(rand(1,cur_end_ind-cur_start_ind+1)-0.5);
        elseif (humacts(i).type == 1 || humacts(i).type == 4) % reach or fail
            % compute hand offest position
            dur_reach = humacts(i).dist_reach/humacts(i).vel_reach;
            dur_mid = dur_reach+humacts(i).dur_draw;
            dur_retreat = humacts(i).dist_reach/humacts(i).vel_retreat;
            dur_total = dur_mid+dur_retreat;
            reach_end_ind = floor(dur_reach * samp_per_sec) + 1;
            mid_end_ind = floor(dur_mid * samp_per_sec) + 1;
            total_end_ind = floor(dur_total * samp_per_sec) + 1;

            reach_dists = nan*ones(1,total_end_ind);
            t1 = linspace(0, dur_reach, reach_end_ind);
            reach_dists(1:reach_end_ind) = humacts(i).vel_reach*t1;
            reach_dists(reach_end_ind+1:mid_end_ind) = humacts(i).dist_reach;
            t2 = linspace(dur_mid, dur_total, total_end_ind-mid_end_ind);
            reach_dists(mid_end_ind+1:total_end_ind) = ...
                humacts(i).dist_reach-humacts(i).vel_retreat*(t2-dur_mid);

            % find the hand offset positions in the detector's frame
            hand_pos_task = [reach_dists; ...
                             zeros(1,numel(reach_dists)); ...
                             ones(1,numel(reach_dists))];
            reach_task_frame = slots(humacts(i).bin_ind).task_frame;
            detector_frame = slots(d_slot).reach_loc;
            hand_pos_det = repmat([detector_off';0],1,numel(reach_dists)) + ...
                                  detector_frame^-1 * reach_task_frame * hand_pos_task;
            
            % the distance to the center of the detector's frame is the detection
            slot_dists(d_slot,cur_end_ind-numel(reach_dists)+1:cur_end_ind) = ...
                sqrt(hand_pos_det(1,:).^2 + hand_pos_det(2,:).^2);
        end
    end
end

for bin_ind = 1:numel(binavail)
    for i = 2:2:numel(binavail{bin_ind})
        if binavail{bin_ind}(i) < samp_start
            continue
        elseif binavail{bin_ind}(i-1) >= samp_end
            break
        end
        avail_start = max(binavail{bin_ind}(i-1), samp_start);
        avail_end = min(binavail{bin_ind}(i), samp_end);
        cur_start_ind = ceil((avail_start - samp_start) * samp_per_sec)+1;
        cur_end_ind = floor((avail_end - samp_start) * samp_per_sec)+1;
        avail_slot = availslot{bin_ind}(i);
        detect_dists(bin_ind,cur_start_ind:cur_end_ind) = ...
            slot_dists(avail_slot,cur_start_ind:cur_end_ind);
    end
end
