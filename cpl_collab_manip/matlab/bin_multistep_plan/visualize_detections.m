function [] = visualize_detections(t, detections_sorted, max_time, num_bins, nowtimesec, bin_names)

hold on
axis([0, max_time, 0, num_bins])
ylabels = {};
maxdetect = max(detections_sorted(:));
for i = 1:num_bins
    plot(t,detections_sorted(i,:)/(1.1*maxdetect)-i+num_bins,'k')
    plot(t,zeros(1,numel(t))-i+num_bins,'k')
    
    ylabels{i} = sprintf('Bin %s', bin_names{num_bins-i+1});
end
plot([nowtimesec, nowtimesec], [0, num_bins], 'g');

AX = gca;
set(AX,'YTick',(1:num_bins)-0.5);
set(AX,'YTickLabel',ylabels);
set(AX,'YGrid','off');
set(AX,'XGrid','on');
