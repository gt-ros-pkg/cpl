function [] = hum_reaches_viz(humacts, binavail, create_fig, time_interval)
if create_fig
    figure(1)
    clf
end

mintime = time_interval(1);
maxtime = time_interval(2);

for i = 1:numel(binavail)
    subplot(numel(binavail),1,i)
    hold on
    xlim([mintime maxtime])
    ylim([0 1.4])
    for j = 2:2:numel(binavail{i})
        start_time = max(binavail{i}(j-1), mintime);
        end_time = min(binavail{i}(j), maxtime);
        plot([start_time end_time], [1.1 1.1], 'b', 'LineWidth', 8)
    end
end

for i = 1:numel(humacts)
    for j = 1:numel(binavail)
        subplot(numel(binavail),1,j)
        if humacts(i).type == 0  % nothing
            plot(humacts(i).time*[1 1], [0 1.3], 'm')
        elseif humacts(i).type == 3 && humacts(i).bin_ind == j  % wait
            start_time = max(mintime, humacts(i-1).time);
            end_time = min(maxtime, humacts(i).time);
            plot(start_time*[1 1], [1.1 1.5], 'c')
            plot([start_time end_time], [1.3 1.3], 'c', 'LineWidth', 8)
            plot([start_time end_time], [0.05 0.05], 'c')
        elseif humacts(i).type == 2 % assembly
            plot(humacts(i).time*[1 1], [1.1 1.5], 'k')
            plot([humacts(i-1).time humacts(i).time], [1.3 1.3], 'k', 'LineWidth', 8)
            t = linspace(humacts(i-1).time, humacts(i).time, 100);
            plot(t, abs(0.1*rand(size(t))), 'k')
        elseif (humacts(i).type == 1 || humacts(i).type == 4) && humacts(i).bin_ind == j 
            % reach or fail
            if humacts(i).type == 1 % reach
                color = 'g';
            else % failed reach
                color = 'r';
            end
            dur_reach = humacts(i).dist_reach/humacts(i).vel_reach;
            dur_mid = dur_reach+humacts(i).dur_draw;
            dur_retreat = humacts(i).dist_reach/humacts(i).vel_retreat;
            dur_total = dur_mid+dur_retreat;
            action_start_time = humacts(i).time - dur_total;
            t1 = linspace(0, dur_reach, 100);
            plot(t1+action_start_time, humacts(i).vel_reach*t1,color)
            t2 = linspace(dur_reach, dur_mid, 100);
            plot(t2+action_start_time, humacts(i).dist_reach+t2*0,color)
            t3 = linspace(dur_mid, dur_total, 100);
            plot(t3+action_start_time, humacts(i).dist_reach-humacts(i).vel_retreat*(t3-dur_mid),color)
            if humacts(i).type == 1 % reach
                plot(humacts(i).time*[1 1], [1.1 1.5], 'g')
                plot([humacts(i-1).time humacts(i).time], [1.3 1.3], 'g', 'LineWidth', 8)
            else % failed reach
                plot(humacts(i).time*[1 1], [1.1 1.5], 'r')
                plot([humacts(i-1).time humacts(i).time], [1.3 1.3], 'r', 'LineWidth', 8)
            end
        end
    end
end
