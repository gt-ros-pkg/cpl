
MODEL_PATH = 'task_linear_chain_robot/model';

DRAW_START_DISTRIBUTION  = {'b1_1', 'b1_2', 'b1_3', 'b1_4', 'b1_5', 'b1_6', 'b2_1', 'b3_1', 'b4_1'};
DRAW_START_DISTRIBUTION  = {'Body', 'body6', 'Nose1', 'Nose2', 'WingTail'};
DRAW_END_DISTRIBUTION    = {};




bin_req.bin_id        = {3, 11, 2, 15};
bin_req.symbol_names1 = {'Body', 'Nose1', 'Nose2', 'WingTail'};
bin_req.symbol_types1 = {'start', 'start', 'start', 'start'};
bin_req.symbol_names2 = {'body6', 'nose12', 'nose21', 'tail_ad4'};
bin_req.symbol_types2 = {'start', 'start', 'start', 'start'};



