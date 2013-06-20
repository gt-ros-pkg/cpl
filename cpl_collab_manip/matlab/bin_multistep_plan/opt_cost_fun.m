function [cost,varargout] = opt_cost_fun(times, slot_states, plan, ...
                                         rm_cost_fns, lt_cost_fns, traj_dur_ind, getplan)
% times(1) is the start of the first action
% all following values are durations of the following actions
% times(1) + times(2) is the start of the second action
action_times = cumsum(times);
cur_slot_states = slot_states;

cost = 0;

for plan_step = 1:numel(action_times)
    plan_act = plan(plan_step);
    act_ind_start = round(action_times(plan_step));
    if plan_act > 0
        % have empty slot, fill it
        bin_arrive_ind = act_ind_start + 2*traj_dur_ind;
        cur_cost = lt_cost_fns(plan_act, bin_arrive_ind);
        for slot_id = 1:numel(slot_states)
            if slot_states(slot_id) == 0
                slot_states(slot_id) = plan_act;
                break;
            end
        end
    else
        % no empty slot, must remove one
        for slot_id = 1:numel(slot_states)
            bin_id = slot_states(slot_id);
            bin_depart_ind = act_ind_start + traj_dur_ind;
            % if this bin was not in the workspace when we started,
            % then it was delivered at some point
            was_delivered = all(bin_id ~= cur_slot_states);
            rm_costs(slot_id) = rm_cost_fns(bin_id, bin_depart_ind);
        end
        % remove bin of least remove cost
        [cur_cost, rm_slot_id] = min(rm_costs);
        plan(plan_step) = -slot_states(rm_slot_id);
        slot_states(rm_slot_id) = 0;
    end
    all_costs(plan_step) = cur_cost;
    cost = cost + cur_cost;
end

if getplan
    varargout{1} = plan;
    varargout{2} = all_costs;
end
