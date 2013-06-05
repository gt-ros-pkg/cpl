function n = k_planning_init( m )
%K_PLANNING_INIT Summary of this function goes here
%   Detailed explanation goes here
%   k: persistent data storage for planning


PORT_NUMBER = 54321;

n = struct;

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

