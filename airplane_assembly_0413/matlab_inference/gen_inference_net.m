

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
m.params.min_duration           = 30;
rate = 30;


duration_mean = 50 / m.params.downsample_ratio;
duration_var  = 400 * m.params.duration_var_scale / m.params.downsample_ratio^2;
m.params.trick.fakedummystep    = nxmakegaussian(m.params.T, duration_mean, duration_var);


% m.params.use_start_conditions = 0;
% m.params.trick.fakedummystep  = NaN;


% %%read from shray's task descriptions and make changes to Nam's model
% task_var = read_task_description('linear_chain_1');
% no_bins_task = numel(task_var);
% %go through various symbols
% for i=1:numel(model.grammar.symbols)
%     %check if terminal
%     if model.grammar.symbols(i).is_terminal == 1
%        temp_bin = model.grammar.symbols(i).detector_id;
%        temp_action_name = strcat('lc1_',model.grammar.symbols(i).name);
%        
%        for j=1:no_bins_task
%           if temp_bin == task_var{j}.bin_id
%              for draw_no = 1:numel(task_var{j}.draw_ids)
%                 if task_var{j}.draw_ids{draw_no} == temp_action_name
%                     %temp_action_name
%                     model.grammar.symbols(i).manual_params.duration_mean = task_var{j}.draw_means(draw_no)*rate;
%                     model.grammar.symbols(i).manual_params.duration_var = (task_var{j}.draw_stds(draw_no)*rate)^2;
%                 end
%                 
%              end
%           end
%        end
%     end
%     
% end
%%done


%

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
        
        if isfield(model.grammar.symbols(m.g(end).id), 'learntparams')
            duration_mean = model.grammar.symbols(m.g(end).id).learntparams.duration_mean / m.params.downsample_ratio;
            duration_var  = m.params.duration_var_scale * model.grammar.symbols(m.g(end).id).learntparams.duration_var / m.params.downsample_ratio^2;
        else
            duration_mean = model.grammar.symbols(m.g(end).id).manual_params.duration_mean / m.params.downsample_ratio;
            duration_var  = m.params.duration_var_scale * model.grammar.symbols(m.g(end).id).manual_params.duration_var / m.params.downsample_ratio^2;
        end
            
            
        duration      = nxmakegaussian(m.params.T, duration_mean, duration_var);
        duration(1:round(m.params.min_duration/m.params.downsample_ratio)) = 0;
        duration      = duration / sum(duration);
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
        
        if isfield(r, 'or_prob')
            m.g(end).orweights = r.or_prob;
        end;
        
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
if 0
    m.g(m.s).start_distribution(30) = 1;
else
    prior_len = m.params.T;
    m.g(m.s).start_distribution(1:prior_len) = 1 / prior_len;
end
m.g(m.s).end_likelihood = ones(1, m.params.T) / m.params.T;

%% set up detection result
m.detection.result = cell(length(m.detection.detectors), 1);
for i=unique([m.g.detector_id])
    m.detection.result{i} = ones(m.params.T);
end;

disp 'Generating inference network is successful'
return;

end

