beam_counts = [3, 3, 2, 1]; % number of branches to make at each depth of the beam search
traj_dur = 3.54/2; % time (s) it takes to complete a robot trajectory

undodur = 2*traj_dur; % time (s) it takes to remove a bin and deliver it back
endedweight = 10; % beam heuristic penalty for the step already ended
notbranchweight = 10; % beam heuristic penalty for not being on the same branch
planning_cycle = 1; % seconds expected till next replan
opt_fun_evals = 100; % max number of optimization function calls
max_time = 200; % maximum time to display in simulation

nowtimeind = round(nowtimesec*rate+1); % t-index of current time
N = numel(probs{1,1});
t = linspace(0,(N-1)/rate,N);
numbins = size(probs,1);
