function n = k_planning_process( n, m, nt, frame_info )
%K_PLANNING_PROCESS Summary of this function goes here
%   Detailed explanation goes here
%   k
%   m: inference structure
%   nt: current timing point
%   frame_info: current world state (bins & hands' position)

%% receive executed plan

if ~isfield(n, 'executedplan')
    n.executedplan.events = [];
end

executedplan = n.executedplan;

if n.ros_tcp_connection.BytesAvailable > 0
    disp receive_executedplan
    len = fread(n.ros_tcp_connection, 1, 'int');
    executedplan = char(fread(n.ros_tcp_connection, len, 'char'))';
    executedplan = nx_fromxmlstr(executedplan);
end


%% kelsey optimization
n.bin_distributions = extract_bin_requirement_distributions( m );

N = length(n.bin_distributions);
    
    
probs       = {};
slot_states = [];
bin_names   = {};
rate        = 30 / m.params.downsample_ratio;
debug       = 1;
nowtimesec  = nt * m.params.downsample_ratio / 30;

for i=1:length(n.bin_distributions)
    
    probs{i,1} = n.bin_distributions(i).bin_needed;
    probs{i,2} = n.bin_distributions(i).bin_nolonger_needed;
    
    bin_names{i} = [binid2name(n.bin_distributions(i).bin_id) '(id ' num2str(n.bin_distributions(i).bin_id) ')'];
    
    b = n.bin_distributions(i).bin_id;
    
    if ~isempty(frame_info.bins(b).H)
        d = norm([-1, -1.3] - [frame_info.bins(b).pq(1), frame_info.bins(b).pq(2)]);
        condition_no = d > 1;
    else
        condition_no = 1;
    end

    if ~condition_no
        slot_states(end+1) = i;
    end
end

assert(length(slot_states) <= 3);
for i=length(slot_states)+1:3
    slot_states(i) = 0;
end;


i = multistep(probs, slot_states, bin_names, nowtimesec, rate, debug);

if i == 0,
    disp nothing_To_do
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

%% send
planning_s     = nx_toxmlstr(bestplan);
fwrite(n.ros_tcp_connection, length(planning_s), 'int'); 
fwrite(n.ros_tcp_connection, planning_s, 'char');

%% save
n.executedplan  = executedplan;
if ~isfield(n, 'bestplans')
    n.bestplans = bestplan;
else
    n.bestplans(end+1) = bestplan;
end

end














