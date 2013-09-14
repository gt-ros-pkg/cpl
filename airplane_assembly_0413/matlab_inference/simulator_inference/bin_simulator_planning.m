function [next_action] = bin_simulator_planning(bin_distributions, nowtimeind, ...
                                                bins_cur_avail, lastrminds, ...
                                                detection_raw_result, rate, debug)

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

    % TODO
    condition_no = ~any(bin_id == bins_cur_avail); % true if bin not in ws

    if ~condition_no
        % multistep arg 2
        slot_states(end+1) = i;
    end
end


% TODO
for i=length(slot_states)+1:3
    slot_states(i) = 0;
end;

% multistep arg 6
% event_hist = [];
% for event = n.executedplan.events
%     start_time = event.matlab_execute_time;
%     if event.matlab_finish_time == -1
%         end_time = nowtimeind;
%     else
%         end_time = event.matlab_finish_time;
%     end
%     bin_ind = event.bin_ind;
%     if event.sname(1) == 'A'
%         remove_mult = 1;
%     else
%         remove_mult = -1;
%     end
%     event_hist(end+1,:) = [remove_mult*bin_ind, start_time, end_time];
% end

% converts action_names_gt -> waiting_times
% multistep arg 7
% waiting_times = [];
% for act_names_ind = 1:numel(n.action_names_gt)
%     cur_act = n.action_names_gt(act_names_ind);
%     if numel(cur_act.name) >= 7 && strcmp(cur_act.name(1:7),'Waiting')
%         start_time = cur_act.start;
%         if act_names_ind >= numel(n.action_names_gt)
%             end_time = nowtimeind;
%         else
%             next_act = n.action_names_gt(act_names_ind+1);
%             end_time = next_act.start;
%         end
%         act_strs = strsplit(cur_act.name, '_')
%         % bin_id = str2num(act_strs{2}(2:end))
%         bin_id = str2num(act_strs{2})
%         bin_ind = find(bin_id == [bin_distributions.bin_id], 1)
%         waiting_times(end+1,:) = [bin_ind, start_time, end_time]
%     end
% end

% TODO:
% arg6: event_hist(end+1,:) = [remove_mult*bin_ind, start_time, end_time];
% arg7: waiting_times(end+1,:) = [bin_ind, start_time, end_time]
event_hist = [];
waiting_times = [];

[next_action, best_plan] = multistep(probs, slot_states, bin_names, nowtimesec, rate, ...
                                     event_hist, waiting_times, lastrminds, debug, detection_raw_result);
if next_action ~= 0
    next_action = sign(next_action)*bin_id_map(abs(next_action));
end
