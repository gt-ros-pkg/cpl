function [] = visualize_bin_activity(bin_seq, times, bin_names, history, ...
                                     slot_states, numbins, rate, tnow, t, maxtime, ...
                                     event_hist, waiting_times, for_humanoids)
bar_width = 100/numbins;
hold on

for i = 1:size(waiting_times,1)
    bin_ind = waiting_times(i,1);
    start_time = t(waiting_times(i,2));
    end_time = t(waiting_times(i,3));
    yval = numbins-(bin_ind-1);
    plot([start_time, end_time], [yval, yval], 'c','LineWidth',bar_width*1.5);
    
end

for i = 1:size(event_hist,1)
    bin_ind = abs(event_hist(i,1));
    is_remove = event_hist(i,1) < 0;
    start_time = event_hist(i,2)/rate;
    end_time = event_hist(i,3)/rate;
    yval = numbins-(bin_ind-1);
    if is_remove
        color = 'r';
        show_text = 'Remove';
    else
        color = 'b';
        show_text = 'Deliver';       
    end
    plot([start_time, end_time], [yval, yval], color,'LineWidth',bar_width*0.8);

    if for_humanoids
        text(max(start_time-1,1), yval+0.3, show_text);
    end
end

for i = 1:size(times,1)
    bin = abs(bin_seq(i));
    isrm = bin_seq(i) < 0;
    bin_ind = bin;
    yval = numbins-(bin_ind-1);
    if isrm
        color = 'y';        
        %if latest_rmv_bins(bin_ind)<0 || ...
         %   abs(latest_rmv_bins(bin_ind)-times(i,1))>20 
        if times(i,1) >tnow+0.3   
            show_text = 'Remove';%strcat('Remove', num2str(tnow), '&', num2str(times(i,1)));
        else
            show_text = '';%num2str(abs(latest_rmv_bins(bin_ind)-times(i,1)));%'taken care of';
        end
    else
        color = 'g';
        %if latest_add_bins(bin_ind)<0 && ...
         %   abs(latest_add_bins(bin_ind)-times(i,1))>20 
        if times(i,1) > tnow+0.3
            show_text = 'Deliver';%strcat('Remove', num2str(tnow), '&', num2str(times(i,1)));%'Deliver';
        else
            show_text = '';%num2str(abs(latest_add_bins(bin_ind)-times(i,1)));%'taken care of';
        end
    end

    plot([times(i,1), times(i,2)], [yval, yval],color,'LineWidth',bar_width*0.8);
    % plot([times(i,1), times(i,2)], [yval, yval],color,'LineWidth',bar_width*0.8);
    if for_humanoids
        text(max(times(i,1)-1,1), max(yval+0.3,1), show_text);
    end
end
plot([tnow, tnow], [0, numbins+1], 'g');
    
for hist_ind = 1:numel(history.nowtimesec_inf)
    for bin_id = history.ws_bins(hist_ind,:)
        if bin_id == 0
            continue
        end
        if hist_ind == 1
            start_time = 0;
        else
            start_time = history.nowtimesec_inf(hist_ind-1);
        end
        end_time = history.nowtimesec_inf(hist_ind);
        yval = numbins-(bin_id-1);
        plot([start_time, end_time], [yval, yval],'m','LineWidth',0.6*bar_width/2);
    end
end

if ~for_humanoids
    for i = 1:numbins
       ylabels{i} = sprintf('Bin %s', bin_names{numbins-i+1});
    end
else    
    text(tnow+0.5, numbins+0.8, sprintf('Now'));
    for i = 1:numbins
       ylabels{i} = sprintf('Bin %d', numbins-i+1);
    end
end

AX = gca;
%maxtime = max(times(:));
axis([0, maxtime, 0, numbins+1])
set(AX,'YTick',(1:numbins));
set(AX,'YTickLabel',ylabels);
box on;
grid on;
hold off;
