clear

% initial environment
bin_init_slots = [1, 4, 5];

% robot current plan
% act_type: 0 = wait, 1 = deliver, 2 = remove
% the times are when the robot action starts
robplan = struct;
robplan.bin_inds =  [   0,    2,    0,    1,   0,    2,    0     1,    0     2,    0 ];
robplan.act_types = [   0,    1,    0,    2,   0,    2,    0     1,    0     1,    0 ];
robplan.times =     [ 0.0, 10.0,  0.0, 22.0, 0.0, 30.0,  0.0  45.0,  0.0  55.0,  0.0 ];

% human's plan
humplan = struct( ...
    'start_time', 5.0, ...
    'durs_step', [ 10.0, 3.0, 4.0, 5.0, 3.0, 5.0, 6.0], ...
    'step_bin',  [    1,   1,   1,   1,   1,   2,   2]);

likelihood_params = struct;
likelihood_params.sigma = 0.10;
likelihood_params.latent_noise = 0.05;
likelihood_params.future_weight = 0.1;

detector_offset = [0, -0.06];

slots = gen_slots();
[robacts, binstates, binavail, availslot] = gen_rob_bin_states(robplan, bin_init_slots, slots);
humacts = gen_reaches(humplan, binavail, availslot, slots);
hum_reaches_viz(humacts, binavail, 1, [0, 70]);
rob_bin_states_viz(binstates, binavail, robacts, slots, [0, 70]);
detector_viz(humacts, slots, binavail, availslot, [0, 50, 70], 2000, ...
             likelihood_params, detector_offset);
wait_stats = fluency_measures(humacts)
