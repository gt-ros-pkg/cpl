

both_sum_waits = cell(2,2);
both_sum_sq_waits = cell(2,2);
both_max_waits = cell(2,2);
both_wait_periods = cell(2,2);
both_failed_reaches = cell(2,2);
both_sum_waits_mat = [];
both_sum_sq_waits_mat = [];
both_max_waits_mat = [];
both_wait_periods_mat = [];
both_failed_reaches_mat = [];

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

cond_order = [1, 1; 2, 1; 2, 2; 1, 2]';
for cond_ind = 1:4
    both_sum_waits_mat(cond_ind,:) = both_sum_waits{cond_order(1,cond_ind),cond_order(2,cond_ind)};
    both_sum_sq_waits_mat(cond_ind,:) = both_sum_sq_waits{cond_order(1,cond_ind),cond_order(2,cond_ind)};
    both_max_waits_mat(cond_ind,:) = both_max_waits{cond_order(1,cond_ind),cond_order(2,cond_ind)};
    both_wait_periods_mat(cond_ind,:) = both_wait_periods{cond_order(1,cond_ind),cond_order(2,cond_ind)};
    both_failed_reaches_mat(cond_ind,:) = both_failed_reaches{cond_order(1,cond_ind),cond_order(2,cond_ind)};
end

x_boxplot_labels = {'{RD, HC}', '{RD, LC}', '{UD, LC}', '{UD, HC}'};
figure(44)
subplot(1,2,1)
boxplot(both_sum_waits_mat', 'whisker', inf);
AX = gca;
set(AX, 'XTick', (1:4));
set(AX, 'XTickLabel', x_boxplot_labels);
set(get(AX, 'YLabel'), 'String', 'Total Wait Time (s)');
subplot(1,2,2)
boxplot(both_sum_sq_waits_mat', 'whisker', inf);
AX = gca;
set(AX, 'XTick', (1:4));
set(AX, 'XTickLabel', x_boxplot_labels);
set(get(AX, 'YLabel'), 'String', 'Sum Squared Wait Time (s^2)');
