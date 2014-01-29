
debug_planning = 1;
create_movie = 1;
extra_info = 0;

if 1
    % icra_complex_task
    init_for_task_linear_chain_7
    m = gen_inference_net(MODEL_PATH);
    m.bin_req = bin_req;

    SEED = 63;
    if 1
        detector_offset = [0.0,0.0];
    else
        detector_offset = [0.0,0.10];
    end
    likelihood_params = struct;
    if 0
        likelihood_params.sigma = 0.03;
        likelihood_params.latent_noise =  0.00005;
        likelihood_params.future_weight = 0.0001;
    else
        likelihood_params.sigma = 0.08;
        likelihood_params.latent_noise =  0.0;
        likelihood_params.future_weight = 0.001;
    end
    bin_names = {'1', '2', '3', '4', '5'};
    bin_id_order = 1:numel(bin_names);

    % acts = be_path; 
    % acts = bf_path; 
    acts = all_acts; 
    % acts = cf_path; 
end

if 0
    % icra_complex_task
    init_for_icra_complex_task
    m = gen_inference_net(MODEL_PATH);
    m.bin_req = bin_req;

    SEED = 63;
    if 1
        detector_offset = [0.0,0.0];
    else
        detector_offset = [0.0,0.10];
    end
    likelihood_params = struct;
    if 1
        likelihood_params.sigma = 0.03;
        likelihood_params.latent_noise =  0.00005;
        likelihood_params.future_weight = 0.0001;
    else
        likelihood_params.sigma = 0.08;
        likelihood_params.latent_noise =  0.0;
        likelihood_params.future_weight = 0.001;
    end
    bin_names = bin_req.symbol_names1;
    bin_id_order = 1:numel(bin_names);

    % acts = be_path; 
    % acts = bf_path; 
    acts = ce_path; 
    % acts = cf_path; 
end

if 0
    % iros_workshop_2chains_task
    init_for_iros_workshop_2chains_task
    m = gen_inference_net(MODEL_PATH);
    m.bin_req = bin_req;

    SEED = 43;
    if 1
        detector_offset = [0.0,0.0];
    else
        detector_offset = [0.0,0.10];
    end
    likelihood_params = struct;
    if 1
        likelihood_params.sigma = 0.03;
        likelihood_params.latent_noise =  0.00005;
        likelihood_params.future_weight = 0.0001;
    else
        likelihood_params.sigma = 0.12;
        likelihood_params.latent_noise =  0.005;
        likelihood_params.future_weight = 0.01;
    end

    body_acts = {'body1', 'body2', 'body3', 'body4', 'body5', 'body6'};
    nosea_acts = {'nose_a1'};
    noseh_acts = {'nose_h1'};
    wingat_acts = {'wing_at1'};
    wingh_acts = {'wing_h1'};
    tailat_acts = {'tail_at1', 'tail_at2', 'tail_at3', 'tail_at4', 'tail_at5', 'tail_at6'};
    tailh_acts = {'tail_h1', 'tail_h2', 'tail_h3', 'tail_h4', 'tail_h5', 'tail_h6'};
    bin_id_order = [3, 11, 10, 12, 7, 14, 13];

    bin_names = {};
    bin_names{1} = 'A';
    for i = 1:3
        bin_names{2*i} = sprintf('B%d',i);
        bin_names{2*i+1} = sprintf('C%d',i);
    end

    % full human action sequence
    % airplane
    % acts = [body_acts, nosea_acts, wingat_acts, tailat_acts]; 
    % helicopter
    acts = [body_acts, noseh_acts, wingh_acts, tailh_acts]; 
end

% set RNG so that results are reproducable
s = RandStream('mt19937ar', 'Seed', SEED);
RandStream.setGlobalStream(s);

% extract duration model from grammar variables
symbols = m.grammar.symbols;
[duration_means, duration_stds, humplan_bin_ids, all_bin_ids] = hum_act_plan_2_dur_model(acts, symbols);
all_bin_ids = bin_id_order;

% initial slot locations for all bins
% first bin is in the workspace, rest are away
bin_init_slots = [1, 4:(2+numel(all_bin_ids))];

% create randomly sampled human action plan
humplan = struct;
humplan.start_time = 2+4*rand();
planlen = numel(humplan_bin_ids);
humplan.durs_step = duration_means + duration_stds.*randn(1,planlen);
for i = 1:numel(all_bin_ids)
    humplan.step_bin(humplan_bin_ids == all_bin_ids(i)) = i;
end

% initializes the robot's execution plan structure
robplan = struct('bin_inds', [], 'act_types', [], 'times', []);

% create the physical locations for all slots
slots = gen_slots();
