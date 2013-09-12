function [] = rob_act_viz(robacts, slots, create_fig, time_interval)
if create_fig
    figure(4)
    clf
end
hold on
mintime = time_interval(1);
maxtime = time_interval(2);
numslots = numel(slots);
has_wait = 0;
for act_ind = 2:numel(robacts)
    prev_slot = robacts(act_ind).prev_slot;
    source_slot = robacts(act_ind).source_slot;
    target_slot = robacts(act_ind).target_slot;
    start_time = robacts(act_ind).start_time;
    rm_time = robacts(act_ind).rm_time;
    dv_time = robacts(act_ind).dv_time;
    end_time = robacts(act_ind).end_time;
    bin_ind = robacts(act_ind).bin_ind;
    if bin_ind < 0
        has_wait = 1; % is a wait
        last_time = max(robacts(act_ind-1).end_time, mintime);
        continue
    end
    prev_yval = numslots-(prev_slot-1);
    source_yval = numslots-(source_slot-1);
    target_yval = numslots-(target_slot-1);
    if has_wait
        plot([last_time, start_time], [prev_yval, prev_yval], 'c','LineWidth',2.0);
        has_wait = 0;
    end
    plot([start_time, rm_time], [prev_yval, source_yval], 'k','LineWidth',2.0);
    plot([rm_time, dv_time], [source_yval, target_yval], 'b','LineWidth',4.0);
    plot([dv_time, end_time], [target_yval, target_yval], 'k','LineWidth',2.0);
    text(rm_time, source_yval+0.3, sprintf('%ds', bin_ind));
    text(dv_time, target_yval+0.3, sprintf('%de', bin_ind));
end
if has_wait
    plot([last_time, maxtime], [target_yval, target_yval], 'c','LineWidth',2.0);
end

AX = gca;
axis([0, maxtime, 0, numslots+1])
set(AX,'YTick',(1:numslots));
ylabels = {};
for i = 1:numslots
   ylabels{i} = sprintf('Slot %d', numslots-i+1);
end
set(AX,'YTickLabel',ylabels);
box on;
grid on;
hold off;
