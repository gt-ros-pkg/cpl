function k = k_planning_process( k, m, nt, frame_info )
%K_PLANNING_PROCESS Summary of this function goes here
%   Detailed explanation goes here
%   k
%   m: inference structure
%   nt: current timing point
%   frame_info: current world state (bins & hands' position)


k.bin_distributions = extract_bin_requirement_distributions( m );


end

