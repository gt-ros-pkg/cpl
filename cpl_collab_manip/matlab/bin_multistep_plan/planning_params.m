% beam_counts = [3, 3, 2, 1]; % number of branches to make at each depth of the beam search
max_beam_depth = 3;
traj_dur = 4.6;%2*3.54/2; % time (s) it takes to complete a robot trajectory
traj_dur_ind = round(rate*traj_dur); % traj_dur in 

% undo_dur = 2*traj_dur; % time (s) it takes to remove a bin and deliver it back
undo_dur = 2*traj_dur; % time (s) it takes to remove a bin and deliver it back
undo_dur_ind = ceil(undo_dur*rate); % time (s) it takes to remove a bin and deliver it back
endedweight = 10; % beam heuristic penalty for the step already ended
notbranchweight = 10; % beam heuristic penalty for not being on the same branch
planning_cycle = 1; % seconds expected till next replan
opt_fun_evals = 200; % max number of optimization function calls
max_time = 200; % maximum time to display in simulation

nowtimeind = round(nowtimesec*rate+1); % t-index of current time
N = numel(probs{1,1});
t = linspace(0,(N-1)/rate,N);
numbins = size(probs,1);

% the minimum relevance for a bin before it's rejected completely from plans
min_bin_relev = -t(end); 
