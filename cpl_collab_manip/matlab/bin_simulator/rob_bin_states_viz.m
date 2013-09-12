function [] = rob_bin_states_viz(binstates, binavail, robacts, slots, time_interval)

figure(6)
clf
subplot(3,1,1)
slots_hist_viz(binstates, slots, 0, time_interval)
subplot(3,1,2)
rob_act_viz(robacts, slots, 0, time_interval)
subplot(3,1,3)
bin_avail_viz(binavail, 0, time_interval)
