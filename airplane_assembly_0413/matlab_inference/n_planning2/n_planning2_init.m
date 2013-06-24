function n = n_planning2_init( m )
%N_PLANNING_INIT Summary of this function goes here
%   Detailed explanation goes here


PORT_NUMBER = 54321;

n = struct;

% return;

%% cache cost functions
T = m.params.T;
n.cache_cost.cost_squareddist    = zeros(2*T + 1, 1);
n.cache_cost.cost_earlyexpensive = zeros(2*T + 1, 1);
n.cache_cost.cost_lateexpensive  = zeros(2*T + 1, 1);
for i=1:2*T + 1
    n.cache_cost.cost_squareddist(i)    = cost_squareddist(i-T-1);
    n.cache_cost.cost_earlyexpensive(i) = cost_earlyexpensive(i-T-1);
    n.cache_cost.cost_lateexpensive(i)  = cost_lateexpensive(i-T-1);
    n.cache_cost.cost_zeros(i)          = cost_zeros(i-T-1);
end


%% connect to ROS node
n.ros_tcp_connection                  = tcpip('localhost', PORT_NUMBER);
n.ros_tcp_connection.OutputBufferSize = 99999;
n.ros_tcp_connection.InputBufferSize  = 99999;
disp('Planner: Try connecting to ROS node....');
while 1
    try
        fopen(n.ros_tcp_connection)
        disp('Planner: Connected');
        break;
    catch e
        pause(1);
    end
end

end

