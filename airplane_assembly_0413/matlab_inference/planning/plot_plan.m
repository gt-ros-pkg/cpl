function  plot_plan( plans, nt)
%PLOT_PLAN Summary of this function goes here
%   Detailed explanation goes here

if ~iscell(plans)
    plot_plan({plans}, nt);
    return;
end

cla;
ylim([2 6]);
grid on;
hold on;

plot([nt nt], [-10 10], 'g');

for k=1:length(plans)
    
    plan = plans{k};
    i    = 0;
    
    for e=plan.events
        i = i + 1;
        
        exe_time = e.optimal_t - e.pre_duration;
        if isfield(e, 'matlab_execute_time')
            exe_time = e.matlab_execute_time;
        end
            
        plot([exe_time exe_time+e.pre_duration], 2*k+[0 0.5], 'color', nxtocolor(e.bin_id));
        plot([exe_time+e.pre_duration exe_time+e.pre_duration+e.post_duration], 2*k+[0.5 0], 'color', nxtocolor(e.bin_id));
        text(double(exe_time) , 2*k+0.5+0.5*mod(i,3), e.sname);
    end

end

hold off;

end

