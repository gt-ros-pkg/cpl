function n = n_planning2_process( n, m, nt, frame_info )
%N_PLANNING_PROCESS Summary of this function goes here
%   Detailed explanation goes here


n.bin_distributions = extract_bin_requirement_distributions( m );

N = length(n.bin_distributions);

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


n.executedplan  = executedplan;

nt = ceil(nt);

%% optimize
test_n_p2

% check plan valid
if length(n.executedplan.events) > 0 & strcmp(n.executedplan.events(end).sname, bestplan.events(1).sname)
    warning 'skip best plan';
    return;
end

%% send
planning_s     = nx_toxmlstr(bestplan);
fwrite(n.ros_tcp_connection, length(planning_s), 'int'); 
fwrite(n.ros_tcp_connection, planning_s, 'char');

%% save
if ~isfield(n, 'bestplans')
    n.bestplans = {bestplan};
else
    n.bestplans{end+1} = bestplan;
end


end

