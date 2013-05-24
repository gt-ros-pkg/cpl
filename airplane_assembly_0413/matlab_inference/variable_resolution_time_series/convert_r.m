
clear;
load ../s/learntdata
load d0


%% convert
tic;

rs = create_resolution_structure(data.T, 10, 1.01);

for i=1:length(m.g)
    if m.g(i).isterminal
        
        m.g(i).likelihood = m.g(i).obv_duration_likelihood ./ m.g(i).durationmat;
        m.g(i).likelihood(find(isnan(m.g(i).likelihood))) = 0;
        
        m.g(i).durationmat = down_sample_vrts_mat(m.g(i).durationmat, rs.csize);
        m.g(i).likelihood  = vrts_downsample_mat_avg(m.g(i).likelihood, rs);
        
        m.g(i).obv_duration_likelihood = m.g(i).durationmat .* m.g(i).likelihood;
    end
end


m.g(m.s).start_distribution = down_sample_vrts(m.g(m.s).start_distribution, rs.csize);
m.g(m.s).end_likelihood     = down_sample_vrts(m.g(m.s).end_likelihood, rs.csize);

toc;

%% inference
data.T = rs.T;
m.T    = rs.T;

data.compute_joint_dist = 0;


m = m_inference_v2(m, data, 1);
data.grammar.symbols = calculate_symbol_distribution(m, data.grammar.symbols);

%% convert back
for i=1:length(data.grammar.symbols)
    
    data.grammar.symbols(i).start_distribution = up_sample_vrts(data.grammar.symbols(i).start_distribution, rs.csize);
    data.grammar.symbols(i).end_distribution = up_sample_vrts(data.grammar.symbols(i).end_distribution, rs.csize);
end

%% draw
figure;
DRAW_START_DISTRIBUTION = {'Body', 'Nose_A', 'Wing_AT', 'Tail_AT'};
DRAW_START_DISTRIBUTION = {'Body', 'wing_at1', 'wing_at2', 'wing_at3', 'tail_at1'};
DRAW_END_DISTRIBUTION   = {};
nt = 1;
plot_distribution;

