function [action, best_plan] = multistep(probs, slot_states, bin_names, ...
                                         nowtimesec, rate, ...
                                         event_hist, waiting_times, ...
                                         lastrminds, ...
                                         debug, detection_raw_result)

if 0
    bin_names{1} = 'A';
    for i = 1:3
        bin_names{2*i} = sprintf('B%d',i+1);
        bin_names{2*i+1} = sprintf('C%d',i+1);
    end
end

planning_params

for i = 1:numbins
    if lastrminds(i) > 0
        % hacky multiplier which discounts the cost of delivery for a while until
        % new information comes in
        info_wait_center = t(lastrminds(i)) + 4.0*t(undo_dur_ind) % time at which function hits 50%
        info_wait_scale = t(undo_dur_ind)/1.2 % seconds after the center until the function hits 90%
        info_wait_mult = atan(3.0*(t-info_wait_center)/info_wait_scale)/pi + 0.5; % sigmoid function
        info_wait_mult(nowtimeind+1:nowtimeind+100)
        for j = 1:2
            future_probs = info_wait_mult(nowtimeind:end).*probs{i,j}(nowtimeind:end);
            lost_weight =  sum(probs{i,j}(nowtimeind:end)) - sum(future_probs);
            probs{i,j}(1:nowtimeind-1) = probs{i,j}(1:nowtimeind-1)* ...
                                         lost_weight/sum(probs{i,j}(1:nowtimeind-1));
            probs{i,j}(nowtimeind:end) = future_probs;
        end
    end
end

% if ~isfield(history, 'slots')
%     history.probs = {};
%     history.slots = [];
%     history.bin_names = bin_names;
%     history.nowtimes = [];
%     history.rate = rate;
%     history.event_hist = {};
%     history.waiting_times = {};
%     history.debug = debug;
%     history.numbins = numbins;
%     history.nowtimesec = [];
%     history.nowtimeind = [];
%     history.max_time = max_time;
%     history.t = t;
%     history.undo_dur = undo_dur;
%     history.undo_dur_ind = undo_dur_ind;
% 
%     history.actions_sorted = {}
%     history.all_plans_sorted = {};
%     history.all_action_starts = {};
%     history.all_action_ends = {};
%     history.costs_sorted = {};
%     history.all_costs_split = {};
%     history.is_delivered = {};
%     history.bin_relevances = {};
% end
% history.probs{end+1} = probs;
% history.slots(end+1,:) = slot_states;
% history.nowtimes(end+1) = nowtimesec;
% history.nowtimesec(end+1) = nowtimesec;
% history.nowtimeind(end+1) = nowtimeind;
% history.event_hist{end+1} = event_hist;
% history.waiting_times{end+1} = waiting_times;

% optimizer printout verbosity
if debug
    display_arg = 'off';
    %display_arg = 'final-detailed';
else
    display_arg = 'off';
end
% opt_options = optimset('Algorithm', 'active-set', 'FinDiffRelStep', 1, 'MaxFunEvals', opt_fun_evals);
% opt_options = optimset('Algorithm', 'active-set', 'FinDiffRelStep', 1, 'MaxFunEvals', opt_fun_evals, ...
% opt_options = optimset('Algorithm', 'interior-point', 'MaxFunEvals', opt_fun_evals, ...
opt_options = optimset('Algorithm', 'interior-point', 'FinDiffRelStep', 1, 'DiffMinChange', 1, 'MaxFunEvals', opt_fun_evals, ...
                       'Display', display_arg);
%opt_options = optimset('Algorithm', 'active-set', 'MaxFunEvals', opt_fun_evals);

% Generate potential sequences of bin deliveries.
% These sequences are based on a beam search through
% bins not in the workspace currently and weighted using a heuristic which prefers bins
% whose expected start time is closer in the future, has not yet ended, and whose
% branch probability is high.
% t              : The full time vector [0..Tend]
% beam_counts    : The number of different bins to consider at each step of the beam search.
%                  If the first number is 3, the search will consider the top 3 bins as the
%                  first step. If the second number is 2, it will consider the top 2 bins
%                  (once the 1st choice is removed).  The length of this vector is the
%                  depth of the delivery sequence.
% probs          : Probability values for the bin step distributions.
% slot_states    : The state of the workspace slots (0 means slot is empty, >0 is the bin ID filling)
% nowtimeind     : The t-index of the current time.
% endedweight    : The penalty weight representing the number of seconds to penalize 
%                  the bin if the probability the bin has ended is 0.5
% notbranchweight: The penalty weight representing the number of seconds to penalize 
%                  the bin if the probability the bin is on this branch is 0.5
bin_relevances = get_bin_relevances(t, probs, slot_states, nowtimeind, endedweight, notbranchweight);
bin_relevances(bin_relevances < min_bin_relev) = -inf;
exit_early = 0;
if all(bin_relevances == -inf)
    % no bins are relevant, robot should just wait
    action = 0;
    best_plan = [];
    exit_early = 1;
else
    deliv_seqs = gen_deliv_seqs(bin_relevances, max_beam_depth);
end

% for bin_ind = 1:numbins
%     if size(event_hist,1) == 0
%         lastrminds(bin_ind) = -1;
%     else
%         bin_removes = find(event_hist(:,1)==-bin_ind);
%         if numel(bin_removes) == 0
%             lastrminds(bin_ind) = -1;
%         else
%             lastrminds(bin_ind) = max(event_hist(bin_removes,3)');
%         end
%     end
% end

% Precompute cost functions for deliver/removal of every bin
for bin_ind = 1:numbins
    binprob = sum(probs{bin_ind,1});
    startprobs = probs{bin_ind,1} / binprob;
    endprobs = probs{bin_ind,2} / binprob;
    if exit_early
        is_delivered(bin_ind) = 0;
    else
        is_delivered(bin_ind) = any(bin_ind == deliv_seqs(1,:));
    end
    rm_cost_fns(bin_ind,:) = remove_cost_precomp(t, startprobs, endprobs, binprob, ...
                                                 undo_dur, is_delivered(bin_ind));
    lt_cost_fns(bin_ind,:) = late_cost_precomp(t, startprobs, endprobs, binprob, ...
                                               nowtimeind, lastrminds(bin_ind), undo_dur_ind);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if exit_early
    if debug
        figure(101)
        clf
        subplot_tight(3,1,1,[.0,0.1]);
        visualize_bin_activity([], [], bin_names, ...
                               [], slot_states, numbins, rate, ...
                               nowtimesec, t, max_time, event_hist, waiting_times, false);
        subplot_tight(3,1,2,[.0,0.1]);
        visualize_bin_probs(t, numbins, probs, bin_names, bin_relevances, ...
                            nowtimesec, nowtimeind, max_time, true);
        % subplot(3,1,3)
        % visualize_cost_funs(t, rm_cost_fns, lt_cost_fns, nowtimesec, max_time);
        
        % subplot(3,1,3)
        subplot_tight(3,1,3,[.05,0.1]);
        visualize_detections(t, detections_sorted, max_time, numbins, nowtimesec, bin_names);
        
        pause(0.05)
        
    elseif debug
        figure(101)
        clf
        subplot(2,1,1)
        visualize_bin_activity([], [], bin_names, ...
                               [], slot_states, numbins, rate, ...
                               nowtimesec, t, max_time, event_hist, waiting_times, true);
        subplot(2,1,2)
        visualize_bin_probs(t, numbins, probs, bin_names, bin_relevances, ...
                            nowtimesec, nowtimeind, max_time, true);
    end
    return
end

all_best_times = [];
all_costs = [];
all_costs_split = [];
all_plans = [];
for i = 1:size(deliv_seqs,1)
    % create a template action plan given a delivery sequence
    plan = create_plan(slot_states, deliv_seqs(i,:));
    % will continue filling slots until they're all filled, then alternate
    % removing a generic bin (determined in optimization) and filling a specified bin
    % in the delivery order

    % create the durations (s) for each action step, given a plan
    % durations = create_durations(plan, traj_dur);
    durations = traj_dur*2 * ones(1,numel(plan));
    
    % optimize the opt_cost_fun for a given plan over the timings each action is completed
    % The solution is bounded below by the start time of the first action, given
    % it is executed now, and the duration of it and subsequent actions, given they execute
    % as soon as the last action is completed
    lower_bounds = [nowtimeind, durations(1:end-1)*rate];

    % the end time of the last action should be before the end of the distribution
    % A = ones(1, numel(lower_bounds));
    % b = numel(t)-durations(end)*rate;

    A = diag(ones(1,numel(durations)-1),-1) + diag(-ones(1,numel(durations)));
    b = -lower_bounds;

    best_cost = inf;
    for start_off = 0:20:40
    % for start_off = 0:10:40
        x_start = cumsum(lower_bounds);
        % x_start(1) = x_start(1) + 40*rate;
        lower_bounds;
        x_start = x_start + start_off*rate;
        x_sol = fmincon(@(x) opt_cost_fun(x, slot_states, plan, ...
                                          rm_cost_fns, lt_cost_fns, ...
                                          traj_dur_ind, nowtimeind, 0), ...
                        x_start, ...
                        A, b, ...
                        [], [], ...
                        [], (numel(t)-10-2*traj_dur_ind)*ones(1,numel(x_start)), ...
                        [], opt_options);
                                          % redeliv_cost_fns, before_prob_fns, during_prob_fns, ...

        % move early delivers to the beginning of the line
        for plan_ind = 1:numel(plan)
            if plan(plan_ind) > 0
                x_sol(plan_ind) = sum(lower_bounds(1:plan_ind));
            else
                break
            end
        end
        % cur_times = cumsum(x_sol / rate);
        cur_times = x_sol / rate;

        % given the optimal timings, find the actual plan and its cost from the optimization function
        % this will fill in the bin removals from the original plan
        [cost, cur_plan, cur_costs] = opt_cost_fun(x_sol, slot_states, plan, rm_cost_fns, lt_cost_fns, traj_dur_ind, nowtimeind, 1);
        % [cost, cur_plan, cur_costs] = opt_cost_fun(x_sol, slot_states, plan, rm_cost_fns, lt_cost_fns, redeliv_cost_fns, before_prob_fns, during_prob_fns, traj_dur_ind, nowtimeind, 1);

        if cost < best_cost
            best_cost = cost;
            best_costs = cur_costs;
            best_times = cur_times;
            best_plan = cur_plan;
        end
    end

    deliver_sequence = deliv_seqs(i,:);
    all_best_times(i,:) = best_times;
    all_costs(i) = best_cost;
    all_costs_split(i,:) = best_costs;
    all_plans(i,:) = best_plan;
end

actions = [];
all_plans_sorted = [];
all_action_starts = [];
all_action_ends = [];
[costs_sorted, cost_inds] = sort(all_costs);
for i = 1:size(deliv_seqs,1)
    ind = cost_inds(i);
    cost = all_costs(ind);
    best_times = all_best_times(ind,:);
    durations = traj_dur*2 * ones(1,numel(plan));
    plan = all_plans(ind,:);
    costs_split = all_costs_split(ind,:);
    action_starts = best_times;
    action_ends = best_times+durations;

    all_plans_sorted(i,:) = plan;
    all_action_starts(i,:) = action_starts;
    all_action_ends(i,:) = action_ends;

    % if action == 0, wait
    % if action  > 0, deliver bin "action"
    % if action  < 0, remove bin "action"
    actions(i) = plan_action(plan, action_starts, nowtimesec, planning_cycle);

    %if 1 && i == 1% debug && i == 1
    if 0 && debug && i == 1
        figure(100+i)
        clf
        % subplot(3,1,1)
        subplot_tight(3,1,1,[.01,0.1]);
        visualize_bin_activity(plan, [action_starts', action_ends'], bin_names, ...
                               [], slot_states, numbins, rate, ...
                               nowtimesec, t, max_time, event_hist, waiting_times,false);
        % subplot(3,1,2)
        subplot_tight(3,1,2,[.01,0.1]);
        visualize_bin_probs(t, numbins, probs, bin_names, bin_relevances, ...
                            nowtimesec, nowtimeind, max_time, false);
        % subplot(4,1,3)
        % visualize_cost_funs(t, rm_cost_fns, lt_cost_fns, nowtimesec, max_time);
        
        % subplot(3,1,3)
        subplot_tight(3,1,3,[.01,0.1]);
        visualize_detections(t, detections_sorted, max_time, numbins, nowtimesec, bin_names);
        
        if actions(i) == 0
            action_name = 'WAIT';
        elseif actions(i) > 0
            action_name = sprintf('DELIVER %s', bin_names{actions(i)});
        else
            action_name = sprintf('REMOVE %s', bin_names{-actions(i)});
        end
        title(sprintf('Cost: %.1f | Action: %s', cost, action_name))
        pause(0.05)
        
    elseif debug && i==1
        figure(100+i)
        clf
        subplot_tight(3,1,1,[.02,0.1]);
        visualize_bin_activity(plan, [action_starts', action_ends'], bin_names, ...
                               [], slot_states, numbins, rate, ...
                               nowtimesec, t, max_time, event_hist, waiting_times, true);

        subplot_tight(3,1,2,[.02,0.1]);
        visualize_bin_probs(t, numbins, probs, bin_names, bin_relevances, ...
                            nowtimesec, nowtimeind, max_time, true);
        
        subplot_tight(3,1,3,[.05,0.1]);
        visualize_detections(t, detections_sorted, max_time, numbins, nowtimesec, bin_names);
        
    end
end

plan_costs = nan*zeros(size(all_plans_sorted,1)*2,size(all_plans_sorted,2)+1);
for i = 1:size(all_plans_sorted,1)
    plan_costs(2*i-1,1) = costs_sorted(i);
    plan_costs(2*i-1,2:end) = all_plans_sorted(i,:);
    plan_costs(2*i,2:end) = all_costs_split(i,:);
end
plan_costs

action = actions(1);
best_plan = [all_plans_sorted(1,:)', all_action_starts(1,:)', all_action_ends(1,:)'];

% history.actions_sorted{end+1} = actions;
% history.all_plans_sorted{end+1} = all_plans_sorted;
% history.all_action_starts{end+1} = all_action_starts;
% history.all_action_ends{end+1} = all_action_ends;
% history.costs_sorted{end+1} = costs_sorted;
% history.all_costs_split{end+1} = all_costs_split;
% history.is_delivered{end+1} = is_delivered;
% history.bin_relevances{end+1} = bin_relevances;
