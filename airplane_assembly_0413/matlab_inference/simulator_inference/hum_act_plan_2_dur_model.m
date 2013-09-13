function [duration_means, duration_stds, bin_ids, all_bin_ids] = hum_act_plan_2_dur_model(acts, symbols)
FPS = 30;

duration_means = nan*ones(1,numel(acts));
duration_stds = nan*ones(1,numel(acts));
bin_ids = nan*ones(1,numel(acts));
all_bin_ids = [];
for i = 1:numel(symbols)
    all_bin_ids = unique([all_bin_ids, symbols(i).detector_id]);
    for j = 1:numel(acts)
        if strcmp(symbols(i).name, acts{j}) == 1
            duration_means(j) = symbols(i).manual_params.duration_mean / FPS;
            duration_stds(j) = sqrt(symbols(i).manual_params.duration_var / FPS^2);
            bin_ids(j) = symbols(i).detector_id;
        end
    end
end
