clear

humplan = struct( ...
    'start_time', 5.0, ...
    'durs_step', [ 10.0, 3.0, 4.0, 5.0, 3.0, 5.0, 6.0], ...
    'step_bin',  [    1,   1,   1,   1,   1,   2,   2]);

binavail{1}  = [-inf, 17, 22, inf];
availslot{1} = [  -1,  1, -1,   3];
binavail{2}  = [-inf, 17, 40, inf];
availslot{2} = [  -1,  2, -1,   1];

slots = gen_slots();
humacts = gen_reaches(humplan, binavail, availslot, slots);
hum_reaches_viz(humacts, binavail, 1);
