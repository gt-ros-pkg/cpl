function  plot_plan( plans)
%PLOT_PLAN Summary of this function goes here
%   Detailed explanation goes here

if ~iscell(plans)
    plot_plan({plans});
    return;
end

cla;
ylim([2 6]);
grid on;
hold on;


for k=1:length(plans)
    
    plan = plans{k};
    i    = 0;
    
    if isfield(plan, 'events') 
    for e=plan.events
        i = i + 1;
        
        exe_time = e.optimal_t - e.pre_duration;
        if isfield(e, 'matlab_execute_time')
            exe_time = e.matlab_execute_time;
        end
        
        end_time = exe_time + e.pre_duration + e.post_duration;
        if isfield(e, 'matlab_finish_time') & e.matlab_finish_time > exe_time
            end_time = e.matlab_finish_time;
        end
        
        opt_time = exe_time + e.pre_duration / (e.pre_duration + e.post_duration) * (end_time - exe_time);
            
        plot([exe_time opt_time], 2*k+[0 0.5], 'color', nxtocolor(e.bin_id));
        plot([opt_time end_time], 2*k+[0.5 0], 'color', nxtocolor(e.bin_id));
        text(double(exe_time) , 2*k+0.5+0.5*mod(i,3), e.sname);
    end
    end
end

hold off;

end

