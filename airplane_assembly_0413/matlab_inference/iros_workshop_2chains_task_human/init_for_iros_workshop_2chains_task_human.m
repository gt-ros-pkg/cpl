
MODEL_PATH = 'iros_workshop_2chains_task_human/model';

%DRAW_START_DISTRIBUTION  = {'Body', 'body1','body6', 'Nose_A', 'Nose_H', 'Tail_AT', 'Tail_H'};
%DRAW_END_DISTRIBUTION    = {'S'};

DRAW_START_DISTRIBUTION  = { 'body1', 'Nose_A', 'Nose_H','Wing_AT', 'Wing_H', 'Tail_AT', 'Tail_H'};
DRAW_END_DISTRIBUTION    = {};


bin_req.bin_id        = {3, 11, 10, 12, 7, 14, 13};
bin_req.bin_id        = {3, 11, 12, 10, 7, 5, 9}; % new bin color
bin_req.symbol_names1 = {'Body', 'Nose_A', 'Nose_H', 'Wing_AT', 'Wing_H', 'Tail_AT', 'Tail_H'};
bin_req.symbol_types1 = {'start', 'start', 'start', 'start', 'start', 'start', 'start'};
bin_req.symbol_names2 = {'body6', 'nose_a1', 'nose_h1', 'wing_at1', 'wing_h1', 'tail_at6', 'tail_h6'};
bin_req.symbol_types2 = {'start', 'start', 'start', 'start', 'start', 'start', 'start'};














