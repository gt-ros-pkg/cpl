function [] = visualize_bin_activity(bin_seq, times, bins, tnow, maxtime)
bar_width = 100/length(bins);

hold on
for i = 1:size(times,1)
    bin = abs(bin_seq(i));
    isrm = bin_seq(i) < 0;
    bin_ind = find(bins==bin);
    yval = numel(bins)-(bin_ind-1);
    if isrm
        color = 'r';
    else
        color = 'b';
    end
    plot([times(i,1), times(i,2)], [yval, yval],color,'LineWidth',bar_width);
end
plot([tnow, tnow], [0, numel(bins)+1], 'g');
for i = 1:numel(bins)
    ylabels{i} = sprintf('Bin %d', bins(numel(bins)-i+1));
end

AX = gca;
%maxtime = max(times(:));
axis([0, maxtime, 0, numel(bins)+1])
set(AX,'YTick',(1:numel(bins)));
set(AX,'YTickLabel',ylabels);
box on;
grid on;
hold off;
