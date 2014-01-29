
MODEL_PATH = 'icra_complex_task/model';

bin_req.bin_id        = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
bin_req.symbol_names1 = {'A1', 'B1', 'B2', 'C1', 'C2', 'C3', 'D1', 'E1', 'F1', 'F2', 'G1'};
bin_req.symbol_types1 = {'start', 'start', 'start', 'start', 'start', 'start', ...
                         'start', 'start', 'start', 'start', 'start'};
bin_req.symbol_names2 = {'a1_t3', 'b1_t1', 'b2_t2', 'c1_t1', 'c2_t1', 'c3_t1', 'd1_t2', 'e1_t2', 'f1_t1', 'f2_t1', 'g1_t1'};
bin_req.symbol_types2 = {'start', 'start', 'start', 'start', 'start', 'start', ...
                         'start', 'start', 'start', 'start', 'start'};

a_acts = {'a1_t1', 'a1_t2', 'a1_t3'};
b_acts = {'b1_t1', 'b2_t1', 'b2_t2'};
c_acts = {'c1_t1', 'c2_t1', 'c3_t1'};
d_acts = {'d1_t1', 'd1_t2'};
e_acts = {'e1_t1', 'e1_t2'};
f_acts = {'f1_t1', 'f2_t1'};
g_acts = {'g1_t1'};
be_path = [a_acts, b_acts, d_acts, e_acts, g_acts];
bf_path = [a_acts, b_acts, d_acts, f_acts, g_acts];
ce_path = [a_acts, c_acts, d_acts, e_acts, g_acts];
cf_path = [a_acts, c_acts, d_acts, f_acts, g_acts];
