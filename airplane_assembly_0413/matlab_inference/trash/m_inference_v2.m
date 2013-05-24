function m = m_inference_v2( m , data , obv_duration_likelihood_computed)
%M_INFERENCE Summary of this function goes here
%   Detailed explanation goes here

    m = compute_null_likelihood(m, m.s);
    
    % compute P(end | start) * P( Z | start, end)
    if ~exist('obv_duration_likelihood_computed') || ~obv_duration_likelihood_computed
        for i=1:length(m.g)
            m.g(i).obv_duration_likelihood = m.g(i).durationmat .* m.g(i).likelihood;
        end
    end
    

    % forward phase
    m.if.g = m.g;
    m = forward_phase(m, m.s, data);
    
    % backward phase
    m.ib.g = m.g;
    m = backward_phase(m, m.s, data);

    % merge forward & backward
    for i=1:length(m.ib.g)
        g = m.ib.g(i);
        g.end_distribution = m.if.g(i).end_distribution .* g.end_likelihood;
        g.end_distribution = g.end_distribution / sum(g.end_distribution);
        g.start_distribution = m.if.g(i).start_distribution .* g.start_likelihood;
        g.start_distribution = g.start_distribution / sum(g.start_distribution);
        m.ib.g(i) = g;
    end
    
    % compute happening prob
    m.if.g(m.s).prob_notnull = 1;
    m.ib.g(m.s).prob_notnull = 1;
    m = compute_prob_notnull(m, m.s);

    % compute symbol distribution
    m.model.grammar.symbols = calculate_symbol_distribution(m, m.model.grammar.symbols);
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function m = compute_null_likelihood( m, gid )

    m.g(gid).or_log_othersnull_likelihood = NaN;

    if m.g(gid).isterminal
        return;
    end
    
    log_null_likelihood = 0;
    
    if ~m.g(gid).isterminal
        
        for i=m.g(gid).prule 
            m = compute_null_likelihood(m,i);
            log_null_likelihood = log_null_likelihood + m.g(i).log_null_likelihood;
        end
        
        
    end

    m.g(gid).log_null_likelihood = log_null_likelihood;
    
    %~~~~~~~~~~~ or
    if ~m.g(gid).andrule
        sum_log_null_likelihood = 0;
        
        for i=1:length(m.g(gid).prule)
            sum_log_null_likelihood = sum_log_null_likelihood + m.g(m.g(gid).prule(i)).log_null_likelihood;
        end
        
        for i=1:length(m.g(gid).prule)
            m.g(m.g(gid).prule(i)).or_orweight = m.g(gid).orweights(i);
            m.g(m.g(gid).prule(i)).or_log_othersnull_likelihood = sum_log_null_likelihood - m.g(m.g(gid).prule(i)).log_null_likelihood;
        end
    end

end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function m = forward_phase( m , gid , data)

    m.if.g(gid).end_distribution    = nan(1, m.T);
    m.if.g(gid).log_pZ              = nan;
    g                               = m.if.g(gid);
    
    %% wait for bin
    %if g.wait_for_bin == 1 & isfield(m, 'bin_available') & ~isnan(m.bin_available(g.bin_id, 1))
    %    g.start_distribution = nx_maxdistribution(g.start_distribution, m.bin_available(g.bin_id,:));
    %   %disp heraherasdfdf
    %end
    
    if g.isterminal
    %% terminal
        p =  g.start_distribution * g.obv_duration_likelihood;
        g.log_pZ = log(sum(p));
        g.end_distribution = p / sum(p);
        
        if m.params.compute_terminal_joint
            g.joint1 = repmat(g.start_distribution', [1 m.T]) .* g.obv_duration_likelihood;
        end
        
    elseif g.andrule
    %% and rule
    
        start_distribution = g.start_distribution;
        g.log_pZ = 0;
        
        for i=1:length(g.prule)

            m.if.g(g.prule(i)).start_distribution = start_distribution;
            m = forward_phase(m, g.prule(i), data);
            
            g.log_pZ = g.log_pZ + m.if.g(g.prule(i)).log_pZ;
            start_distribution = m.if.g(g.prule(i)).end_distribution;
        end
        
        
        g.end_distribution = start_distribution;
        
    
    else   
    %% or rule 
        for i=1:length(g.prule)
            m.if.g(g.prule(i)).start_distribution = g.start_distribution;
            m = forward_phase(m, g.prule(i), data);
        end
        
        
        % 
        g.end_distribution = zeros(1, m.T);
        for i=1:length(g.prule)
            g.end_distribution = g.end_distribution + ...
                m.if.g(g.prule(i)).or_orweight * ...
                exp(m.if.g(g.prule(i)).log_pZ + m.if.g(g.prule(i)).or_log_othersnull_likelihood) * ...
                m.if.g(g.prule(i)).end_distribution;
        end
        g.log_pZ = log(sum(g.end_distribution));
        g.end_distribution = g.end_distribution / sum(g.end_distribution);
    end

    %m.if.g = nx_assign_struct(m.if.g, gid, g);
    m.if.g(gid) = g;

end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function m = backward_phase( m, gid , data)


    m.ib.g(gid).start_likelihood    = nan;
    m.ib.g(gid).joint_likelihood    = nan;
    m.ib.g(gid).start_distribution  = nan;
    m.ib.g(gid).end_distribution    = nan;
    g                               = m.ib.g(gid);
    

    %% terminal
    if g.isterminal
    
        g.start_likelihood = (g.obv_duration_likelihood * g.end_likelihood')';
        
        if m.params.compute_terminal_joint
            g.joint2 = g.obv_duration_likelihood .* repmat(g.end_likelihood, [m.T 1]);
        end
        
    elseif g.andrule
    %% and rule
    
        end_likelihood = g.end_likelihood;
        
        for i=g.prule(end:-1:1)
            
            m.ib.g(i).end_likelihood = end_likelihood;
            m = backward_phase(m, i, data);
            end_likelihood = m.ib.g(i).start_likelihood;
        end
        
        g.start_likelihood = end_likelihood;
        
    else  %% or rule  
    
        
        for i=g.prule
            
            m.ib.g(i).end_likelihood = g.end_likelihood;
            m = backward_phase(m, i, data);
            
        end
        
        g.start_likelihood = zeros(1, m.T);
        for i=g.prule
            g.start_likelihood = g.start_likelihood  + ...
                m.ib.g(i).or_orweight * ...
                exp(m.ib.g(i).or_log_othersnull_likelihood) * ...
                m.ib.g(i).start_likelihood;
        end
    end

    
    %% wait for bin
    %if g.wait_for_bin == 1 & isfield(m, 'bin_available') & ~isnan(m.bin_available(g.bin_id, 1))
    %    g.start_likelihood = nx_maxdistribution_backward(g.start_likelihood);
    %end
    
    %%
    m.ib.g(gid) = g;
    % m.ib.g = nx_assign_struct(m.ib.g, gid, g);

end



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function m = compute_prob_notnull( m, gid )


    if m.g(gid).isterminal
        return;
        
        
    elseif m.g(gid).andrule
        
        for i=m.g(gid).prule

            m.if.g(i).prob_notnull = m.if.g(gid).prob_notnull;
            
            m.ib.g(i).prob_notnull = m.ib.g(gid).prob_notnull;
            
            m = compute_prob_notnull(m, i);
            
        end
    
    else
        
        s = [];
        
        for i=m.g(gid).prule
            
            log_notnull = log(m.if.g(i).or_orweight) + m.if.g(i).log_pZ + m.if.g(i).or_log_othersnull_likelihood - m.if.g(gid).log_pZ;
            m.if.g(i).prob_notnull = m.if.g(gid).prob_notnull * exp(log_notnull);
            
            s(end+1) = m.if.g(i).prob_notnull * ...
                sum(m.ib.g(i).end_likelihood .* m.if.g(i).end_distribution);
            
        end
        
        s = m.if.g(gid).prob_notnull * s / sum(s);
        
        for i=m.g(gid).prule
            
            m.ib.g(i).prob_notnull = s(1);
            s(1) = [];
            
            m = compute_prob_notnull(m, i);
            
        end
        
    end
    
end

































