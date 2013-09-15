% function [] = detector_viz(humacts, slots, binavail, availslot, samp_interval, samp_num, likelihood_params, detector_off)

% initial environment
bin_init_slots = [1, 2, 3];

% robot current plan
% act_type: 0 = wait, 1 = deliver, 2 = remove
% the times are when the robot action starts
robplan = struct;
robplan.bin_inds =  [];
robplan.act_types = [];
robplan.times =     [];

% human's plan
humplan = struct( ...
    'start_time', 5.0, ...
    'durs_step', [  5.0, 5.0, 5.0], ...
    'step_bin',  [    1,   2,   3]);

[robacts, binstates, binavail, availslot] = gen_rob_bin_states(robplan, bin_init_slots, slots);
humacts = gen_reaches(humplan, binavail, availslot, slots);
detector_viz(humacts, slots, binavail, availslot, [0, 40, 50], 2000, ...
             likelihood_params, detector_offset);

