
clear; clc; load s/learntdata

m   = struct;
m.T = data.T;
m.g = [];

x   = [1];
xm  = [1];
i   = 0;
while 1
    i = i + 1;
    if length(x) < i, break; end;
    
    s = data.grammar.symbols(x(i));
    r = data.grammar.rules(find([data.grammar.rules.left] == x(i)));
    
    m.g(end+1).id           = x(i);
    m.g(end).isterminal     = s.is_terminal;
    m.g(end).bin_id         = s.bin_id;
    m.g(end).wait_for_bin   = s.wait_for_bin;
    
    if s.is_terminal
        
        duration_mean = data.grammar.symbols(m.g(end).id).learntparams.duration_mean / data.downsample_ratio;
        duration_var  = data.duration_var_scale * data.grammar.symbols(m.g(end).id).learntparams.duration_var / data.downsample_ratio^2;
        duration      = nxmakegaussian(m.T, duration_mean, duration_var);
        durationmat   = zeros(m.T,m.T);
        
        for j=1:m.T
            durationmat(j,j:end) = duration(1:m.T-j+1);
        end
        
        m.g(end).durationmat = durationmat;
        m.g(end).likelihood = triu(ones(m.T, m.T));
        m.g(end).log_null_likelihood = log(1);
        
        m.g(end).obv_duration_likelihood = m.g(end).likelihood .* m.g(end).durationmat;
        
    else
        
        m.g(end).prule = r.right; % todo
        m.g(end).prule = length(x) + [1:length(r.right)];
        m.g(end).andrule = ~r.or_rule;
        m.g(end).orweights = r.or_prob;
        
        x = [x r.right];
    end
    
    
end

m.s = 1;
m.g(1).start_distribution = 0 * ones(1, data.T) / data.T;
m.g(1).start_distribution(1:10) = 1 / 10;
m.g(1).end_likelihood = ones(1, data.T) / data.T;

clearvars -except data m;

save s/learntdata
