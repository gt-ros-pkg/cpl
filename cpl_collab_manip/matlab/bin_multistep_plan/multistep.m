function [action, best_plan] = multistep(probs, slot_states, bins_history, bin_names, nowtimesec, rate, debug)

planning_params

if debug
    display_arg = 'final-detailed';
else
    display_arg = 'off';
end
% opt_options = optimset('Algorithm', 'active-set', 'FinDiffRelStep', 1, 'MaxFunEvals', opt_fun_evals);
opt_options = optimset('Algorithm', 'active-set', 'DiffMinChange', 1, 'MaxFunEvals', opt_fun_evals, ...
                       'Display', display_arg);
%opt_options = optimset('Algorithm', 'active-set', 'MaxFunEvals', opt_fun_evals);

% Generate potential sequences of bin deliveries.
bin_relevances = get_bin_relevances(t, probs, slot_states, nowtimeind, endedweight, notbranchweight);
deliv_seqs = gen_deliv_seqs(bin_relevances, beam_counts);
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

all_best_times = [];
all_costs = [];
all_plans = [];
for i = 1:size(deliv_seqs,1)
    % create a template action plan given a delivery sequence
    plan = create_plan(slot_states, deliv_seqs(i,:));
    % will continue filling slots until they're all filled, then alternate
    % removing a generic bin (determined in optimization) and filling a specified bin
    % in the delivery order

    % create the durations (s) for each action step, given a plan
    durations = create_durations(plan, traj_dur);
    
    % optimize the opt_cost_fun for a given plan over the timings each action is completed
    % The solution is bounded below by the completion time of the first action, given
    % it is executed now, and the duration of subsequent actions, given they execute
    % as soon as the last action is completed
    lower_bounds = durations*rate;
    lower_bounds(1) = lower_bounds(1) + nowtimeind;
    A = ones(numel(lower_bounds));
    b = numel(t)*ones(numel(lower_bounds),1);
    x_sol = fmincon(@(x) opt_cost_fun(x, slot_states, plan, t, probs, undodur, nowtimeind, 0), ...
                                      lower_bounds, ...
                                      A, b, ...
                                      [], [], ...
                                      lower_bounds, [], ...
                                      [], opt_options);
    best_times = cumsum(x_sol / rate);

    % given the optimal timings, find the actual plan and its cost from the optimization function
    % this will fill in the bin removals from the original plan
    [cost, fullplan] = opt_cost_fun(x_sol, slot_states, plan, t, probs, undodur, nowtimeind, 1);

    deliver_sequence = deliv_seqs(i,:);
    all_best_times(i,:) = best_times;
    all_costs(i) = cost;
    all_plans(i,:) = fullplan;
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
    durations = create_durations(plan, traj_dur);
    plan = all_plans(ind,:);
    action_starts = best_times-durations;
    action_ends = best_times;

    all_plans_sorted(i,:) = plan;
    all_action_starts(i,:) = action_starts;
    all_action_ends(i,:) = action_ends;

    % if action == 0, wait
    % if action  > 0, deliver bin "action"
    % if action  < 0, remove bin "action"
    actions(i) = plan_action(plan, action_starts, nowtimesec, planning_cycle);

    if debug && i == 1
        figure(100+i)
        clf
        subplot(2,1,1)
        visualize_bin_activity(plan, [action_starts', action_ends'], bin_names, ...
                               slot_states, numbins, nowtimesec, max_time);
        subplot(2,1,2)
        visualize_bin_probs(t, numbins, probs, bin_names, bin_relevances, ...
                            nowtimesec, nowtimeind, max_time);
        if actions(i) == 0
            action_name = 'WAIT';
        elseif actions(i) > 0
            action_name = sprintf('DELIVER %s', bin_names{actions(i)});
        else
            action_name = sprintf('REMOVE %s', bin_names{-actions(i)});
        end
        title(sprintf('Cost: %.1f | Action: %s', cost, action_name))
    end
end

action = actions(1);
best_plan = [all_plans_sorted(1,:)', all_action_starts(1,:)', all_action_ends(1,:)'];
