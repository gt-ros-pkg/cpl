function n = n_planning2_terminate( n )
%N_PLANNING2_TERMINATE Summary of this function goes here
%   Detailed explanation goes here


% close ros tcp connection
fclose(n.ros_tcp_connection);

end

