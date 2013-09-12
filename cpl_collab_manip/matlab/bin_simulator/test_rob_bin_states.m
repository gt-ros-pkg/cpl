clear

bin_init_slots = [1, 4, 5];
% d 4, w, r 1, r 4
robplan = struct;
robplan.bin_inds =  [   0,    2,    0,    1,   0,    2,    0 ];
% act_type: 0 = wait, 1 = deliver, 2 = remove
robplan.act_types = [   0,    1,    0,    2,   0,    2,    0 ];
robplan.times =     [ 0.0, 10.0,  0.0, 25.0, 0.0, 40.0,  0.0 ];

slots = gen_slots();
[robacts, binstates, binavail, availslot] = gen_rob_bin_states(robplan, bin_init_slots, slots);
rob_bin_states_viz(binstates, binavail, robacts, slots);
