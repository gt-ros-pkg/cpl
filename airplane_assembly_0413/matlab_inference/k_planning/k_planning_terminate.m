function k = k_planning_terminate( k )
%K_PLANNING_TERMINATE Summary of this function goes here
%   Detailed explanation goes here

% close ros tcp connection
fclose(k.ros_tcp_connection);

end

