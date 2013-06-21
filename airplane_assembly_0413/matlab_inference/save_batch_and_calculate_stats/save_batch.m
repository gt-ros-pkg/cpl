
SAVE_BATH_FILE_NAME = 'save_batch_and_calculate_stats/a1.mat';

% load old data
try
    load(SAVE_BATH_FILE_NAME);
end

% new one
if ~exist('batch_data')
    batch_data = struct;
    batch_data.id = 1;
else
    batch_data(end+1).id = length(batch_data) + 1;
end

% add
batch_data(end).action_name_gt = action_names_gt;
batch_data(end).executedplan   = k.executedplan;
batch_data(end).multistep_history = k.multistep_history;


% save
save(SAVE_BATH_FILE_NAME, 'batch_data');
clearvars batch_data

















