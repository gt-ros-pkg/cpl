function [] = slots_hist_viz(binstates, slots, create_fig, time_interval)
if create_fig
    figure(3)
    clf
end
hold on
mintime = time_interval(1);
maxtime = time_interval(2);
numslots = numel(slots);
for bin_ind = 1:numel(binstates)
    for time_ind = 2:numel(binstates{bin_ind})
        slot_ind = binstates{bin_ind}(time_ind).slot;
        if slot_ind < 0
            continue
        end
        start_time = binstates{bin_ind}(time_ind-1).time;
        end_time = binstates{bin_ind}(time_ind).time;
        yval = numslots-(slot_ind-1);
        % if is_remove
        %     color = 'r';
        % else
        %     color = 'b';
        % end
        start_time = max(start_time, mintime);
        end_time = min(end_time, maxtime);
        color = 'b';
        plot([start_time, end_time], [yval, yval], color,'LineWidth',8.0);
        plot([start_time, start_time], [yval-0.3, yval+0.3], color);
        plot([end_time, end_time], [yval-0.3, yval+0.3], color);
        text(start_time, yval+0.3, sprintf('%ds', bin_ind));
        text(end_time, yval+0.3, sprintf('%de', bin_ind));
    end
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
