function [] = visualize_bin_activity(best_plan, bin_names, N, rate, nowtimesec, max_time, ...
                                     robacts, humacts, ws_slots, for_humanoids)
numbins = numel(bin_names);

bar_width = 100/numbins;
hold on

% for i = 1:size(waiting_times,1)
%     bin_ind = waiting_times(i,1);
%     start_time = t(waiting_times(i,2));
%     end_time = t(waiting_times(i,3));
for i = 1:numel(humacts)
    if humacts(i).type == 3 
        start_time = max(0, humacts(i-1).time);
        end_time = min(nowtimesec, humacts(i).time);
        bin_ind = humacts(i).bin_ind;
        if start_time > nowtimesec
            break
        end

        yval = numbins-(bin_ind-1);
        plot([start_time, end_time], [yval, yval], 'c','LineWidth',bar_width*1.5);
    end
end

for act_ind = 2:numel(robacts)
    bin_ind = robacts(act_ind).bin_ind;
    if bin_ind <= 0
        continue
    end
    start_time = robacts(act_ind).start_time;
    end_time = robacts(act_ind).end_time;
    source_slot = robacts(act_ind).source_slot;
    target_slot = robacts(act_ind).target_slot;
    if ismember(target_slot, ws_slots) && ~ismember(source_slot, ws_slots)
        is_remove = 0; % deliver
    elseif ~ismember(target_slot, ws_slots) && ismember(source_slot, ws_slots)
        is_remove = 1; % remove
    else
        % shuffle
        'SHUFFLE SHOULD NOT HAPPEN'
        return
    end
    % rm_time = robacts(act_ind).rm_time;
    % dv_time = robacts(act_ind).dv_time;
    if is_remove
        color = 'r';
        show_text = 'Remove';
    else
        color = 'b';
        show_text = 'Deliver';       
    end
    yval = numbins-(bin_ind-1);
    plot([start_time, end_time], [yval, yval], color,'LineWidth',bar_width*0.8);

    if for_humanoids
        text(max(start_time-1,1), yval+0.45, show_text);
    end
end

for i = 1:size(best_plan,1)
    bin_action = best_plan(i,1);
    act_start = best_plan(i,2);
    act_end = best_plan(i,3);
    bin = abs(bin_action);
    isrm = bin_action < 0;
    bin_ind = bin;
    yval = numbins-(bin_ind-1);
    if isrm
        color = 'r';        
        %if latest_rmv_bins(bin_ind)<0 || ...
         %   abs(latest_rmv_bins(bin_ind)-act_start)>20 
        if act_start >nowtimesec+0.3   
            show_text = 'Remove';%strcat('Remove', num2str(nowtimesec), '&', num2str(act_start));
        else
            show_text = '';%num2str(abs(latest_rmv_bins(bin_ind)-act_start));%'taken care of';
        end
    else
        color = 'b';
        %if latest_add_bins(bin_ind)<0 && ...
         %   abs(latest_add_bins(bin_ind)-act_start)>20 
        if act_start > nowtimesec+0.3
            show_text = 'Deliver';%strcat('Remove', num2str(nowtimesec), '&', num2str(act_start));%'Deliver';
        else
            show_text = '';%num2str(abs(latest_add_bins(bin_ind)-act_start));%'taken care of';
        end
    end

    plot([act_start, act_end], [yval, yval],color,'LineWidth',bar_width*0.8);
    % plot([act_start, act_end], [yval, yval],color,'LineWidth',bar_width*0.8);
    if for_humanoids
        text(max(act_start-1,1), max(yval+0.8,1), show_text);
    end
end
plot([nowtimesec, nowtimesec], [0, numbins+1], 'g');
    
% for hist_ind = 1:numel(history.nowtimesec_inf)
%     for bin_id = history.ws_bins(hist_ind,:)
%         if bin_id == 0
%             continue
%         end
%         if hist_ind == 1
%             start_time = 0;
%         else
%             start_time = history.nowtimesec_inf(hist_ind-1);
%         end
%         end_time = history.nowtimesec_inf(hist_ind);
%         yval = numbins-(bin_id-1);
%         plot([start_time, end_time], [yval, yval],'m','LineWidth',0.6*bar_width/2);
%     end
% end

if ~for_humanoids
    for i = 1:numbins
       ylabels{i} = sprintf('Bin %s', bin_names{numbins-i+1});
    end
else    
    text(nowtimesec+0.5, numbins+0.6, sprintf('Now'));
    for i = 1:numbins
       ylabels{i} = sprintf('Bin %s', bin_names{numbins-i+1});
       % ylabels{i} = sprintf('Bin %d', numbins-i+1);
    end
end

AX = gca;
axis([0, max_time, 0, numbins+1])
set(AX,'YTick',(1:numbins));
set(AX,'YTickLabel',ylabels);
set(get(AX, 'XLabel'), 'String', 'Time (s)');
set(get(AX, 'YLabel'), 'String', 'Robot Action Schedule');
box on;
grid on;
hold off;
