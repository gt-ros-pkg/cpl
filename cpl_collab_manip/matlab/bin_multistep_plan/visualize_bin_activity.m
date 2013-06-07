function [] = visualize_bin_activity(bin_seq, times, bin_names, history, ...
                                     slot_states, numbins, rate, tnow, maxtime)
bar_width = 100/numbins;

hold on
% for i = 1:numbins
%     yval = numbins-(i-1);
%     if any(i == slot_states)
%         plot([tnow-5, tnow], [yval, yval],'g','LineWidth',bar_width/2);
%     end
% end
    
for hist_ind = 1:numel(history.nowtimes)
    for bin_id = history.slots(hist_ind,:)
        if bin_id == 0
            continue
        end
        if hist_ind == 1
            start_time = 0;
        else
            start_time = history.nowtimes(hist_ind-1);
        end
        end_time = history.nowtimes(hist_ind);
        yval = numbins-(bin_id-1);
        plot([start_time, end_time], [yval, yval],'m','LineWidth',bar_width/2);
    end
end

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
    plot([times(i,1), times(i,2)], [yval, yval],color,'LineWidth',bar_width);
end
plot([tnow, tnow], [0, numbins+1], 'g');
for i = 1:numbins
    ylabels{i} = sprintf('Bin %s', bin_names{numbins-i+1});
end

AX = gca;
%maxtime = max(times(:));
axis([0, maxtime, 0, numbins+1])
set(AX,'YTick',(1:numbins));
set(AX,'YTickLabel',ylabels);
box on;
grid on;
hold off;
