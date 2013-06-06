function [durations] = create_durations(plan, traj_dur)

remove_dur = traj_dur;
deliver_dur = 3*traj_dur;

durations = zeros(1,numel(plan));

for i = 1:numel(plan)
    plan_act = plan(i);
    if plan_act > 0
        durations(i) = deliver_dur;
    else
        durations(i) = remove_dur;
    end
end
