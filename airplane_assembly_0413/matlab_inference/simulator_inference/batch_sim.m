
if 1
    body_acts = {'body1', 'body2', 'body3', 'body4', 'body5', 'body6'};
    nosea_acts = {'nose_a1'};
    noseh_acts = {'nose_h1'};
    wingat_acts = {'wing_at1'};
    wingh_acts = {'wing_h1'};
    tailat_acts = {'tail_at1', 'tail_at2', 'tail_at3', 'tail_at4', 'tail_at5', 'tail_at6'};
    tailh_acts = {'tail_h1', 'tail_h2', 'tail_h3', 'tail_h4', 'tail_h5', 'tail_h6'};
    bin_id_order = [3, 11, 10, 12, 7, 14, 13];
end

start_seed = 11;
num_runs = 10;
% batch_results = cell(2,2,2);
for run_ind = 1:num_runs
    for conf_ind = 1:2
        for detect_ind = 1:2
            for task_ind = 1:2
                SEED = start_seed + run_ind-1;
                if detect_ind == 1
                    detector_offset = [0.0,0.0];
                else
                    detector_offset = [0.0,0.10];
                end
                likelihood_params = struct;
                if conf_ind == 1
                    likelihood_params.sigma = 0.03;
                    likelihood_params.latent_noise =  0.00005;
                    likelihood_params.future_weight = 0.0001;
                else
                    likelihood_params.sigma = 0.12;
                    likelihood_params.latent_noise =  0.005;
                    likelihood_params.future_weight = 0.01;
                end

                % full human action sequence
                if task_ind == 1
                    % airplane
                    acts = [body_acts, nosea_acts, wingat_acts, tailat_acts]; 
                else
                    % helicopter
                    acts = [body_acts, noseh_acts, wingh_acts, tailh_acts]; 
                end

                bin_sim_planning_main

                if numel(batch_results{conf_ind,detect_ind,task_ind}) == 0
                    batch_results{conf_ind,detect_ind,task_ind} = wait_stats;
                else
                    batch_results{conf_ind,detect_ind,task_ind}(end+1) = wait_stats;
                end
            end
        end
    end
end
