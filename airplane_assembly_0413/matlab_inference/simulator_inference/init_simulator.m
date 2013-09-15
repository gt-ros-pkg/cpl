
debug_planning = 0;
create_movie = 0;
viz_extra_info = 1;

if 0
    SEED = 43;
    if 0
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
