

function m = gen_inference_net(model)

if isstr(model)
    load(model);
end

disp ==========================================================
disp 'Generate inference network'
disp ==========================================================

m = struct;

if ~exist('model')
    disp 'variable model not found. Please load model first'
    return;
end

m                               = model;
m.params.T                      = 1000;
m.params.wait_for_bin           = 1;
m.params.compute_terminal_joint = 0;
m.params.downsample_ratio       = 7;
m.params.duration_var_scale     = 3;
m.params.use_start_conditions   = 1;


duration_mean = 50 / m.params.downsample_ratio;
duration_var  = 400 * m.params.duration_var_scale / m.params.downsample_ratio^2;
m.params.trick.fakedummystep    = nxmakegaussian(m.params.T, duration_mean, duration_var);
% m.params.trick.fakedummystep    = NaN;


m.start_conditions              = ones(length(model.grammar.symbols), m.params.T);

m.g = [];


%% roll out
x   = [1];
xm  = [1];
i   = 0;
while 1
    i = i + 1;
    if length(x) < i, break; end;
    
    s = model.grammar.symbols(x(i));
    r = model.grammar.rules(find([model.grammar.rules.left] == x(i)));
    
    m.g(end+1).id        	= x(i);
    m.g(end).is_terminal  	= s.is_terminal;
    m.g(end).detector_id   	= s.detector_id;
    %m.g(end).wait_for_bin   = s.wait_for_bin;
    
    if s.is_terminal
        
        duration_mean = model.grammar.symbols(m.g(end).id).learntparams.duration_mean / m.params.downsample_ratio;
        duration_var  = m.params.duration_var_scale * model.grammar.symbols(m.g(end).id).learntparams.duration_var / m.params.downsample_ratio^2;
        duration      = nxmakegaussian(m.params.T, duration_mean, duration_var);
        durationmat   = zeros(m.params.T,m.params.T);
        
        for j=1:m.params.T
            durationmat(j,j:end) = duration(1:m.params.T-j+1);
        end
        
        m.g(end).durationmat = durationmat;
        %m.g(end).likelihood = triu(ones(m.params.T, m.params.T));
        m.g(end).log_null_likelihood = log(1);
        
        %m.g(end).obv_duration_likelihood = m.g(end).likelihood .* m.g(end).durationmat;
        m.g(end).obv_duration_likelihood = nan(m.params.T);
    else
        
        m.g(end).prule = r.right; % todo
        m.g(end).prule = length(x) + [1:length(r.right)];
        m.g(end).andrule = ~r.or_rule;
        m.g(end).orweights = r.or_prob;
        
        x = [x r.right];
    end
    

end

%% set up inference struct
for i=1:length(m.g)
    m.g(i).i_forward  = struct;
    m.g(i).i_backward = struct;
    m.g(i).i_final    = struct;
end

%% set up root
m.s =  m.grammar.starting;
m.g(m.s).start_distribution = 0 * ones(1, m.params.T) / m.params.T;
m.g(m.s).start_distribution(1:200) = 1 / 200;
m.g(m.s).end_likelihood = ones(1, m.params.T) / m.params.T;

%% set up detection result
for i=unique([m.g.detector_id])
    m.detection.result{i} = ones(m.params.T);
end;

disp 'Generating inference network is successful'
return;

end

