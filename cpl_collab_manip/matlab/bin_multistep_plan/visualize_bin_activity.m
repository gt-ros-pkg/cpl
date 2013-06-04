function [] = visualize_bin_activity(bin_seq, times, numbins, tnow, maxtime)
bar_width = 100/numbins;

hold on
for i = 1:size(times,1)
    bin = abs(bin_seq(i));
    isrm = bin_seq(i) < 0;
    bin_ind = bin;
    yval = numbins-(bin_ind-1);
    if isrm
        color = 'r';
    else
        color = 'b';
    end
    plot([times(i,1), times(i,2)], [yval, yval],color,'LineWidth',bar_width);
end
plot([tnow, tnow], [0, numbins+1], 'g');
for i = 1:numbins
    ylabels{i} = sprintf('Bin %d', numbins-i+1);
end

AX = gca;
%maxtime = max(times(:));
axis([0, maxtime, 0, numbins+1])
set(AX,'YTick',(1:numbins));
set(AX,'YTickLabel',ylabels);
box on;
grid on;
hold off;
