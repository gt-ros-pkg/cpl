function [measures] = fluency_measures(humacts)

act_times = [humacts.time];
act_types = [humacts.type];

wo_fail_times = act_times(act_types ~= 4);
wo_fail_types = act_types(act_types ~= 4);

wait_inds = find(act_types == 3);
wait_starts = act_times(wait_inds-1);
wait_ends = act_times(wait_inds);
wait_durs = wait_ends - wait_starts;

measures = struct;
measures.sum_wait = sum(wait_durs);
measures.sum_sq_wait = sum(wait_durs.^2);
if numel(wait_durs) > 0
    measures.max_wait = max(wait_durs);
else
    measures.max_wait = 0.0;
end
measures.num_wait_periods = numel(wait_durs);
measures.num_failed_reaches = sum(act_types == 4);
measures.wait_durs = wait_durs;
