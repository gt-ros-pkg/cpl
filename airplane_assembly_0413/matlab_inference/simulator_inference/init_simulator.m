
SEED = 44;
% detector_offset = [0.0,0.0];
detector_offset = [0.0,0.06];
likelihood_params = struct;
likelihood_params.sigma = 0.06;
likelihood_params.latent_noise = 0.005;
likelihood_params.future_weight = 0.01;

% different bin action sets 
% body_acts = {'body1', 'body2', 'body3', 'body4', 'body5', 'body6'};
% nosea_acts = {'nose_a1', 'nose_a2', 'nose_a3'};
% noseh_acts = {'nose_h1', 'nose_h2', 'nose_h3'};
% wingat_acts = {'wing_at1', 'wing_at2', 'wing_at3'};
% wingh_acts = {'wing_h1', 'wing_h2', 'wing_h3', 'wing_h4', 'wing_h5', 'wing_h6'};
% tailat_acts = {'tail_at1', 'tail_at2', 'tail_at3'};
% tailh_acts = {'tail_h1', 'tail_h2', 'tail_h3', 'tail_h4', 'tail_h5', 'tail_h6'};

body_acts = {'body1', 'body2', 'body3', 'body4', 'body5', 'body6'};
nosea_acts = {'nose_a1'};
noseh_acts = {'nose_h1'};
wingat_acts = {'wing_at1'};
wingh_acts = {'wing_h1'};
tailat_acts = {'tail_at1', 'tail_at2', 'tail_at3', 'tail_at4', 'tail_at5', 'tail_at6'};
tailh_acts = {'tail_h1', 'tail_h2', 'tail_h3', 'tail_h4', 'tail_h5', 'tail_h6'};
% full human action sequence
% airplane
acts = [body_acts, nosea_acts, wingat_acts, tailat_acts]; 
% helicopter
% acts = [body_acts, noseh_acts, wingh_acts, tailh_acts]; 

% set RNG so that results are reproducable
s = RandStream('mt19937ar', 'Seed', SEED);
RandStream.setGlobalStream(s);

% extract duration model from grammar variables
symbols = m.grammar.symbols;
[duration_means, duration_stds, humplan_bin_ids, all_bin_ids] = hum_act_plan_2_dur_model(acts, symbols);

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
