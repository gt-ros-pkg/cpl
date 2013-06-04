function [cost,varargout] = opt_cost_fun(times, slot_states, plan, t, probs, undodur, nowtimeind, getplan)
% times(1) is the start of the first action
% all following values are durations of the following actions
% times(1) + times(2) is the start of the second action
action_times = cumsum(times);

cost = 0;

for plan_step = 1:numel(action_times)
    plan_act = plan(plan_step);
    act_time = round(action_times(plan_step));
    if plan_act > 0
        % have empty slot, fill it
        binprob = sum(probs{plan_act,1});
        startprobs = probs{plan_act,1} / binprob;
        endprobs = probs{plan_act,2} / binprob;
        cur_cost = late_cost(t, act_time, startprobs, endprobs, binprob, nowtimeind);
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
            binprob = sum(probs{bin_id,1});
            startprobs = probs{bin_id,1} / binprob;
            endprobs = probs{bin_id,2} / binprob;
            rm_costs(slot_id) = remove_cost(t, act_time, startprobs, endprobs, binprob, undodur);
        end
        % remove bin of least remove cost
        [cur_cost, rm_slot_id] = min(rm_costs);
        plan(plan_step) = -slot_states(rm_slot_id);
        slot_states(rm_slot_id) = 0;
    end
    cur_cost;
    cost = cost + cur_cost;
end

if getplan
    varargout{1} = plan;
end
