

both_sum_waits = cell(2,2);
both_sum_sq_waits = cell(2,2);
both_max_waits = cell(2,2);
both_wait_periods = cell(2,2);
both_failed_reaches = cell(2,2);

for conf_ind = 1:2
    for detect_ind = 1:2
        for task_ind = 1:2
            cond_stats = batch_results{conf_ind,detect_ind,task_ind};
            sum_waits = [cond_stats.sum_wait];
            sum_sq_waits = [cond_stats.sum_sq_wait];
            max_waits = [cond_stats.max_wait];
            all_num_wait_periods = [cond_stats.num_wait_periods];
            all_num_failed_reaches = [cond_stats.num_failed_reaches];
            % cond_stats.wait_durs
            both_sum_waits{conf_ind,detect_ind} = [both_sum_waits{conf_ind,detect_ind}, sum_waits];
            both_sum_sq_waits{conf_ind,detect_ind} = [both_sum_sq_waits{conf_ind,detect_ind}, sum_sq_waits];
            both_max_waits{conf_ind,detect_ind} = [both_max_waits{conf_ind,detect_ind}, max_waits];
            both_wait_periods{conf_ind,detect_ind} = [both_wait_periods{conf_ind,detect_ind}, all_num_wait_periods];
            both_failed_reaches{conf_ind,detect_ind} = [both_failed_reaches{conf_ind,detect_ind}, all_num_failed_reaches];
        end
        mean_both_sum_waits(conf_ind,detect_ind) = mean(both_sum_waits{conf_ind,detect_ind});
        mean_both_sum_sq_waits(conf_ind,detect_ind) = mean(both_sum_sq_waits{conf_ind,detect_ind});
        mean_both_max_waits(conf_ind,detect_ind) = mean(both_max_waits{conf_ind,detect_ind});
        mean_both_wait_periods(conf_ind,detect_ind) = mean(both_wait_periods{conf_ind,detect_ind});
        mean_both_failed_reaches(conf_ind,detect_ind) = mean(both_failed_reaches{conf_ind,detect_ind});
    end
end


