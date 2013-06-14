function plan = gen_random_plan( bin_distributions )
%GEN_RANDOM_PLAN Summary of this function goes here
%   Detailed explanation goes here

    N = length(bin_distributions);

    plan.actions = struct('bin_id', {}, 'action_str', {}, 'distribution', {});
    
    bin_available_num     = sum([bin_distributions.bin_available]);
    BIN_AVAILABLE_NUM_MAX = 3;
    
    no_more_add = 0;
    
    for i=1:100
        
        % gen possible actions
        possible_actions = struct('bin_id', {}, 'action_str', {}, 'distribution', {});
        for j=1:N
            action        = struct;
            action.bin_id = bin_distributions(j).bin_id;
            action.bin_id = j;
            
            if bin_distributions(j).bin_available & (bin_available_num == BIN_AVAILABLE_NUM_MAX | no_more_add)
                action.action_str   = 'Remove';
                action.distribution = bin_distributions(j).bin_nolonger_needed;
            elseif ~bin_distributions(j).bin_available & bin_available_num < BIN_AVAILABLE_NUM_MAX & bin_distributions(j).prob_not_finished > 0.01
                action.action_str   = 'Add';
                action.distribution = bin_distributions(j).bin_needed;
            else
                continue;
            end
            
            
            
            % check exist
            action_exist = 0;
            for a = plan.actions
                if a.bin_id == action.bin_id & strcmp(a.action_str, action.action_str)
                    action_exist = 1;
                end
                if a.bin_id == action.bin_id & strcmp(a.action_str, 'Remove')
                    action_exist = 1;
                end
            end
            if action_exist
                continue;
            end
            
            % add to possible
            possible_actions(end+1) = action;
                
        end
        
        % check to turn on no_more_add, or end
        if length(possible_actions) == 0,
            if ~no_more_add
                no_more_add = 1;
            else
                break;
            end
        end
        
        % pick action
        action   = struct;
        action.t = inf;
        for a = possible_actions
            
            d       = a.distribution;
            a_prob  = sum(d);
            
            d       = d / sum(d);
            a.t     = ceil(sum(d .* [1:length(d)]));
            
            if rand >a_prob * 3
                if strcmp(a.action_str, 'Remove')
                    a.t = a.t - length(d);
                elseif strcmp(a.action_str, 'Add')
                    a.t = a.t + length(d);
                end
            end
            
            if action.t == inf || action.t > a.t
                action = a;
            end
        end
        if action.t == inf
            continue;
        end
        
        % update bin available
        if strcmp(action.action_str, 'Add')
            
            bin_distributions(action.bin_id).bin_available = 1;
            bin_available_num = bin_available_num + 1;
            
        elseif strcmp(action.action_str, 'Remove')
            
            bin_distributions(action.bin_id).bin_available = 0;
            bin_available_num = bin_available_num - 1;
        end
        
        % add to plan
        action = rmfield(action, 't');
        plan.actions(end+1) = action;
        
    end
    

end

function action = gen_random_action(bin_distributions)


end