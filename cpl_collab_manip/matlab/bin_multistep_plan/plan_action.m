function [action] = plan_action(plan, action_starts, tnow, planning_cycle)

if action_starts(1) - tnow > planning_cycle
    action = 0; % wait
else
    action = plan(1); % do immediately
end
