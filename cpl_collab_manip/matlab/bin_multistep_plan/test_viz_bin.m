
bin_seq = [3, 2, -1, 4, 5, -2];
times = [5, 15;
         17, 19;
         21, 40;
         40, 45;
         50, 70;
         75, 95];
 
figure(33)
clf
visualize_bin_activity(bin_seq, times, 1:6, 20, 100)
