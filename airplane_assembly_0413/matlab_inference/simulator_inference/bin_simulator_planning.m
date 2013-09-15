function [next_action] = bin_simulator_planning(bin_distributions, nowtimeind, ...
                                                bins_cur_avail, lastrminds, ...
                                                robacts, humacts, ws_slots, ...
                                                detection_raw_result, rate, debug, extra_info)

probs       = {};
slot_states = [];
bin_names   = {};
nowtimesec  = nowtimeind / rate; % multistep arg 4

bin_id_map = ones(1,length(bin_distributions));
for i=1:length(bin_distributions)
    
    % multistep arg 1
    probs{i,1} = bin_distributions(i).bin_needed;
    probs{i,2} = bin_distributions(i).bin_nolonger_needed;
    
    % multistep arg 3
    bin_names{i} = [binid2name(bin_distributions(i).bin_id) ...
                    '(id ' num2str(bin_distributions(i).bin_id) ')'];
    
    bin_id = bin_distributions(i).bin_id;
    bin_id_map(i) = bin_id;
    detections_sorted(i,:) = detection_raw_result(bin_id,:);

    % TODO
    condition_no = ~any(bin_id == bins_cur_avail); % true if bin not in ws

    if ~condition_no
        % multistep arg 2
        slot_states(end+1) = i;
    end
end

slot_states(end+1:numel(ws_slots)) = 0;

[next_action, best_plan] = multistep(probs, slot_states, bin_names, nowtimesec, rate, ...
                                     robacts, humacts, ws_slots, lastrminds, ...
                                     debug, detections_sorted, extra_info);
if next_action ~= 0
    next_action = sign(next_action)*bin_id_map(abs(next_action));
end
