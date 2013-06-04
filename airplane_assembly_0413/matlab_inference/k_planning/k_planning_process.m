function k = k_planning_process( k, m, nt, frame_info )
%K_PLANNING_PROCESS Summary of this function goes here
%   Detailed explanation goes here
%   k
%   m: inference structure
%   nt: current timing point
%   frame_info: current world state (bins & hands' position)


%% kelsey optimization
k.bin_distributions = extract_bin_requirement_distributions( m );

    N = length(k.bin_distributions);
    
    
    for i=1:1000
        i1 = randi([1 N]);
        i2 = randi([1 N]);
        t = k.bin_distributions(i1);
        k.bin_distributions(i1) = k.bin_distributions(i2);
        k.bin_distributions(i2) = t;
    end
    
    
probs       = {};
bins        = [];
slot_states = [];
rate        = 30 / m.params.downsample_ratio;
debug       = 0;
nowtimesec  = nt * m.params.downsample_ratio / 30;

for i=1:length(k.bin_distributions)
    
    probs{i,1} = k.bin_distributions.bin_needed;
    probs{i,2} = k.bin_distributions.bin_nolonger_needed;
    
    bins(i) = k.bin_distributions(i).bin_id;
    bins(i) = i;
    
end

for b=1:length(frame_info.bins)

    if ~isempty(frame_info.bins(b).H)
        d = norm([-1, -1.3] - [frame_info.bins(b).pq(1), frame_info.bins(b).pq(2)]);
        condition_no = d > 1;
    else
        condition_no = 1;
    end

    if ~condition_no
        slot_states(end+1) = b;
    end

end
assert(length(slot_states) <= 3);
for i=length(slot_states)+1:3
    slot_states(i) = 0;
end;
slot_states = [0 0 0];


i = multistep(probs, bins, slot_states, nowtimesec, rate, debug);

if i == 0,
    disp nothing_To_do
    return;
end

action.a = nxifelse(i > 0, 'add', 'remove');
action.bin_id = k.bin_distributions(abs(i)).bin_id

%% exe

disp([action.a '   bin  ' num2str(action.bin_id)]);

end














