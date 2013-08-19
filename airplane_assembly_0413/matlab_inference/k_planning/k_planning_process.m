function n = k_planning_process( n, m, nt, frame_info , bins_availability, bins_cur_avail, debug, detection_raw_result)
%K_PLANNING_PROCESS Summary of this function goes here
%   Detailed explanation goes here
%   k
%   m: inference structure
%   nt: current timing point
%   frame_info: current world state (bins & hands' position)



n.bin_distributions = extract_bin_requirement_distributions( m );

N = length(n.bin_distributions);
    

%% receive executed plan

if ~isfield(n, 'executedplan')
    n.executedplan.events = [];
end

executedplan = n.executedplan;

while n.ros_tcp_connection.BytesAvailable > 0
    disp receive_executedplan
    len = fread(n.ros_tcp_connection, 1, 'int');
    executedplan = char(fread(n.ros_tcp_connection, len, 'char'))';
    executedplan = nx_fromxmlstr(executedplan);
end


n.executedplan  = executedplan;

nt = ceil(nt);

% check robot moving
robot_moving = 0;
if 1
    if isfield(n, 'executedplan') & isfield(n.executedplan, 'events') & length(n.executedplan.events) > 0
        if n.executedplan.events(end).matlab_finish_time < 0
            % disp 'Robot moving, skip planning';
            % return;
            robot_moving = 1;
        else
            robot_moving = 0;
        end
    end
end



%% create bin history

bins_history = struct([]);
for i=1:length(n.bin_distributions)
    
    bin_id = n.bin_distributions(i).bin_id;
    
    bins_history(i).start = [];
    bins_history(i).end   = [];

    for t=1:size(bins_availability,2)
        
        if bins_availability(bin_id,t) > 0
            
            if t == 1 || ~(bins_availability(bin_id,t-1) > 0)
            	bins_history(i).start(end+1) = t;
            end
            
            if t == size(bins_availability,2) || ~(bins_availability(bin_id,t+1) > 0)
            	bins_history(i).end(end+1) = t;
            end
        end
        
    end
end



%% kelsey optimization
    
probs       = {};
slot_states = [];
bin_names   = {};
rate        = 30 / m.params.downsample_ratio;
nowtimesec  = nt * m.params.downsample_ratio / 30;

for i=1:length(n.bin_distributions)
    
    probs{i,1} = n.bin_distributions(i).bin_needed;
    probs{i,2} = n.bin_distributions(i).bin_nolonger_needed;
    
    bin_names{i} = [binid2name(n.bin_distributions(i).bin_id) '(id ' num2str(n.bin_distributions(i).bin_id) ')'];
    
    b = n.bin_distributions(i).bin_id;
    
    % if ~isempty(frame_info.bins(b).H)
    %     d = norm([-1, -1.3] - [frame_info.bins(b).pq(1), frame_info.bins(b).pq(2)]);
    %     condition_no = d > 1;
    % else
    %     condition_no = 1;
    % end
    condition_no = ~any(b == bins_cur_avail); % true if bin not in ws

    if ~condition_no
        slot_states(end+1) = i;
    end
end

assert(length(slot_states) <= 3);
for i=length(slot_states)+1:3
    slot_states(i) = 0;
end;


ws_bins_inds = sort(slot_states);
n.multistep_history.ws_bins(end+1,:) = ws_bins_inds;
n.multistep_history.nowtimesec_inf(end+1) = nt * m.params.downsample_ratio / 30;

event_hist = [];
for event = n.executedplan.events
    start_time = event.matlab_execute_time;
    if event.matlab_finish_time == -1
        end_time = nt;
    else
        end_time = event.matlab_finish_time;
    end
    bin_ind = event.bin_ind;
    if event.sname(1) == 'A'
        remove_mult = 1;
    else
        remove_mult = -1;
    end
    event_hist(end+1,:) = [remove_mult*bin_ind, start_time, end_time];
end

waiting_times = [];
for act_names_ind = 1:numel(n.action_names_gt)
    cur_act = n.action_names_gt(act_names_ind);
    if numel(cur_act.name) >= 7 && strcmp(cur_act.name(1:7),'Waiting')
        start_time = cur_act.start;
        if act_names_ind >= numel(n.action_names_gt)
            end_time = nt;
        else
            next_act = n.action_names_gt(act_names_ind+1);
            end_time = next_act.start;
        end
        act_strs = strsplit(cur_act.name, '_')
        % bin_id = str2num(act_strs{2}(2:end))
        bin_id = str2num(act_strs{2})
        bin_ind = find(bin_id == [n.bin_distributions.bin_id], 1)
        waiting_times(end+1,:) = [bin_ind, start_time, end_time]
    end
end

[i, best_plan, n.multistep_history] = multistep(probs, slot_states, bin_names, nowtimesec, rate, ...
                                                event_hist, waiting_times, ...
                                                n.multistep_history, debug, detection_raw_result);


if i == 0 || robot_moving
    if robot_moving
        disp('Robot moving, do nothing')
    else
        disp nothing_To_do
    end
    return;
end

action.a = nxifelse(i > 0, 'Add', 'Remove');
action.bin_id = n.bin_distributions(abs(i)).bin_id;

%% viz
% n.bin_is_in(nt,:) = 0;
% for i=1:3
%     if slot_states(i) > 0
%         n.bin_is_in(nt, n.bin_distributions(slot_states(i)).bin_id) = 1;
%     end
% end
% figure(214);
% imagesc(n.bin_is_in);

%% create plan

disp([action.a '   bin  ' num2str(action.bin_id)]);

bestplan                        = executedplan;
bestplan.events                 = [];
bestplan.events(end+1).bin_id   = action.bin_id;
bestplan.events(end).signature  = -1;
bestplan.events(end).name       = 'x';
bestplan.events(end).sname      = [action.a '   bin  ' num2str(action.bin_id)];

bestplan.events(end).pre_duration  = 0;
bestplan.events(end).post_duration = 0;
bestplan.events(end).optimal_t     = nt;
bestplan.events(end).bin_ind       = find(action.bin_id == [n.bin_distributions.bin_id], 1);

%% send
planning_s     = nx_toxmlstr(bestplan);
fwrite(n.ros_tcp_connection, length(planning_s), 'int'); 
fwrite(n.ros_tcp_connection, planning_s, 'char');

%% save
n.executedplan  = executedplan;

if ~isfield(n, 'bestplans')
    
    n.bestplans = {};

end
n.bestplans{end+1} = bestplan;

end














