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

while n.ros_tcp_connection.BytesAvailable > 0
    disp receive_executedplan
    len = fread(n.ros_tcp_connection, 1, 'int');
    executedplan = char(fread(n.ros_tcp_connection, len, 'char'))';
    executedplan = nx_fromxmlstr(executedplan);
end


n.executedplan  = executedplan;

nt = ceil(nt);


% check robot moving
if 0
    if isfield(n, 'executedplan') & isfield(n.executedplan, 'events') & length(n.executedplan.events) > 0
        if n.executedplan.events(end).matlab_finish_time < 0
            disp 'Robot moving, skip planning';
            return;
        end
    end
end


%% optimize
test_n_p2

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

