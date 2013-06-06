function n = n_planning_process2( n, m, nt, frame_info )
%N_PLANNING_PROCESS Summary of this function goes here
%   Detailed explanation goes here

    bin_requirements_distributions = extract_bin_requirement_distributions(m);
    
    
    N = length(bin_requirements_distributions);
    
    
    for i=1:1000
        i1 = randi([1 N]);
        i2 = randi([1 N]);
        t = bin_requirements_distributions(i1);
        bin_requirements_distributions(i1) = bin_requirements_distributions(i2);
        bin_requirements_distributions(i2) = t;
    end
    
    
    for i=1:N
        
        bin_requirements_distributions(i).bin_needed_cost = ef2(bin_requirements_distributions(i).bin_needed, n.cache_cost.cost_lateexpensive);
        
        bin_requirements_distributions(i).bin_nolonger_needed_cost = ef2(bin_requirements_distributions(i).bin_nolonger_needed, n.cache_cost.cost_earlyexpensive);
        
    end
    
    add_bin_id    = [bin_requirements_distributions.bin_id];
    remove_bin_id = [bin_requirements_distributions.bin_id];

    % construct the plan
    planx    = struct;
    planx.t0 = 10;
    
    for i=1:N
        
        planx.events(i).signature       = -1;
        planx.events(i).name            = 'what';
        planx.events(i).sname           = ['add ' num2str(bin_requirements_distributions(i).bin_id)];
        planx.events(i).bin_id          = bin_requirements_distributions(i).bin_id;
        planx.events(i).cost_type       = 'cost_lateexpensive';
        planx.events(i).pre_duration    = 11;
        planx.events(i).post_duration   = 11;
        planx.events(i).distribution    = bin_requirements_distributions(i).bin_needed;
    end
    for i=1:N
        
        planx.events(N+i).signature       = -1;
        planx.events(N+i).name            = 'what';
        planx.events(N+i).sname           = ['remove ' num2str(bin_requirements_distributions(i).bin_id)];
        planx.events(N+i).bin_id          = bin_requirements_distributions(i).bin_id;
        planx.events(N+i).cost_type       = 'cost_lateexpensive';
        planx.events(N+i).pre_duration    = 11;
        planx.events(N+i).post_duration   = 11;
        planx.events(N+i).distribution    = bin_requirements_distributions(i).bin_nolonger_needed;
        
        % swap to
        if i <= N-3
            swapto               = 2+2*i;
            planx.events = [planx.events(1:swapto-1) planx.events(N+i) planx.events(swapto:N+i-1)];
        end
    end
    
    
    %
    planx.mintotalcost = inf;
    best_plan = planx;
    i1 = -1;
    
    for i=1:1000
    %while 1   
        % change plan
        planx = best_plan;
        if ~isinf(planx.mintotalcost)
            
%             % find swap id
%             if i1 == -1,
%                 i1 = 1;
%             else
%                 i1 = i1+1;
%             end
%             i2 = -1;
%             while i2 == -1
%                 for i3=i1+1:2*N
%                     if planx.events(i1).sname(1) == planx.events(i3).sname(1)
%                         i2 = i3;
%                         break;
%                     end
%                 end
%                 if i2 == -1
%                     i1 = i1 + 1;
%                 end
%                 if i1 > 2 * N
%                     break;
%                 end
%             end
%             if i1 > 2 * N
%                 disp Done;
%                 break;
%             end




            i1 = -1;
            i2 = -1;
            while i1 == -1 || i2 == -1 || i1 == i2  || planx.events(i1).sname(1) ~= planx.events(i2).sname(1)
                i1 = randi([1 2*N]);
                i2 = randi([1 2*N]);
            end
            
            % now swap
            t = planx.events(i1);
            planx.events(i1) = planx.events(i2);
            planx.events(i2) = t;
        end
        
        % optimize
        planx = find_optimal_timing_for_order(planx, m, n);
        %disp(planx.mintotalcost);
        
        % evaluate
        if planx.mintotalcost < best_plan.mintotalcost
            best_plan = planx;
            
            if i1 > 0
                disp(['swap ' planx.events(i1).sname ' and ' planx.events(i2).sname ]);
                i1 = -1;
            end
        end
        
        
    end
    
    disp(best_plan.mintotalcost);
    plot_plan(best_plan, 1);
    gogo = 1;
end

