
clc;

SKIP_FIRST_WAITING = 1;

file_name = 'LL_new';

load(file_name)


for b=1:length(batch_data) 
    
    %assert(strcmp(b.action_name_gt(end).name, 'Complete'));
    
    batch_data(b).task_time           = 0;
    batch_data(b).total_wait_time     = 0;
    batch_data(b).longest_wait_time   = 0;
    batch_data(b).total_cost          = 0;
    batch_data(b).wait_times          = [];
    
    for i=1:length(batch_data(b).action_name_gt)-1
        if length(batch_data(b).action_name_gt(i).name) >= 7 & strcmp( batch_data(b).action_name_gt(i).name(1:7), 'Waiting')
            batch_data(b).wait_times(end+1) = batch_data(b).action_name_gt(i+1).start - batch_data(b).action_name_gt(i).start;
        end
    end
    
    if SKIP_FIRST_WAITING
        batch_data(b).wait_times(1) = [];
    end
    
    batch_data(b).total_wait_time     = sum(batch_data(b).wait_times);
    batch_data(b).longest_wait_time   = max(batch_data(b).wait_times);
end

disp(file_name)
disp(['total_wait_time: ' num2str(mean([batch_data.total_wait_time])    * 7 / 30)]);
disp(['longest_wait_time: ' num2str(mean([batch_data.longest_wait_time])  * 7 / 30)]);

