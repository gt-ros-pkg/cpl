interval = [0,200];
hum_reaches_viz(humacts, binavail, 1, interval);
rob_bin_states_viz(binstates, binavail, robacts, slots, interval);
detector_viz(humacts, slots, binavail, availslot, [interval, interval(2)], 2000, ...
             likelihood_params, detector_offset);
