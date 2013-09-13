
body_acts = {'body1', 'body2', 'body3', 'body4', 'body5', 'body6'};
nosea_acts = {'nose_a1', 'nose_a2', 'nose_a3'};
noseh_acts = {'nose_h1', 'nose_h2', 'nose_h3'};
wingat_acts = {'wing_at1', 'wing_at2', 'wing_at3'};
wingh_acts = {'wing_h1', 'wing_h2', 'wing_h3', 'wing_h4', 'wing_h5', 'wing_h6'};
tailat_acts = {'tail_at1', 'tail_at2', 'tail_at3'};
tailh_acts = {'tail_h1', 'tail_h2', 'tail_h3', 'tail_h4', 'tail_h5', 'tail_h6'};
acts = [body_acts, nosea_acts, wingat_acts, tailat_acts];
symbols = m.grammar.symbols;

[duration_means, duration_stds, bin_ids, all_bin_ids] = hum_act_plan_2_dur_model(acts, symbols);
SEED = 44;
s = RandStream('mt19937ar', 'Seed', SEED);
RandStream.setGlobalStream(s);
rand_humplan = struct;
rand_humplan.start_time = 2+4*rand();
planlen = numel(bin_ids);
rand_humplan.durs_step = duration_means + duration_stds.*randn(1,planlen);
for i = 1:numel(all_bin_ids)
    rand_humplan.step_bin(bin_ids == all_bin_ids(i)) = i;
end
