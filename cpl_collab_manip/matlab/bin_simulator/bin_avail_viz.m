function [] = bin_avail_viz(binavail, create_fig, time_interval)
if create_fig
    figure(5)
    clf
end
hold on
mintime = time_interval(1);
maxtime = time_interval(2);
numbins = numel(binavail);
for bin_ind = 1:numbins
    for time_ind = 2:2:numel(binavail{bin_ind})
        start_time = binavail{bin_ind}(time_ind-1);
        end_time = binavail{bin_ind}(time_ind);
        yval = numbins-(bin_ind-1);
        start_time = max(start_time, mintime);
        end_time = min(end_time, maxtime);
        plot([start_time, end_time], [yval, yval], 'b','LineWidth',8.0);
    end
end

AX = gca;
axis([0, maxtime, 0, numbins+1])
set(AX,'YTick',(1:numbins));
ylabels = {};
for i = 1:numbins
   ylabels{i} = sprintf('Bin %d', numbins-i+1);
end
set(AX,'YTickLabel',ylabels);
box on;
grid on;
hold off;
