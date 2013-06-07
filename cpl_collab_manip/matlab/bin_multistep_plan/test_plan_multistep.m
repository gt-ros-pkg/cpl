
t_duration = 400; % number of seconds in distributions
rate = 10; % time points / sec

% gaussian distributions for bin steps
% the last parameter weights the distribution by the probability
% the person is on this branch
% (will only display 50% of t_duration in visualization)
%
% delay is the time between the end of the last step and the beginning of this step
% duration is the time between the start of this step and the end
%              delay, beg sig, duration, end sig, bin prob,
distribs = [       2,       4,       30,      4,      1.00;
                   2,       4,       30,      4,      1.00;
                   2,       4,       30,      4,      1.00;
                   2,       4,       30,      4,      1.00;
                   2,       4,       30,      4,      1.00;
                   2,       4,       30,      4,      1.00;
                   2,       4,       30,      4,      1.00;
];
% bins = 1:7; % bin IDs
bin_names = {};
for i = 1:size(distribs,1)
    bin_names{i} = sprintf('Bin %d', i);
end
slot_states = [1, 0, 0]; % state of workspace slots (0 if empty, >0 if bin ID occupies)
nowtimesec = 20; % time (s) of current time

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
probs = gen_data(distribs, rate, t_duration);
action = multistep(probs, slot_states, [], bin_names, nowtimesec, rate, 1)
