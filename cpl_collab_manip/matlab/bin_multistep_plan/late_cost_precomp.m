function [costs] = late_cost_precomp(t, startprobs, endprobs, binprob, nowtimeind, lastrmind, undo_dur_ind)

% recentdur = round(undo_dur_ind);
% ended_recently_delay_mult = 8;
% prob_ended_recently = sum(endprobs(max(nowtimeind-recentdur,1):nowtimeind));
% nowtimeind_delayed = nowtimeind + round(find(t>=ended_recently_delay_mult,1)*prob_ended_recently);
% prob_end_after_now = sum(endprobs(min(nowtimeind_delayed+1,numel(endprobs)):end));

prob_end_after_now = sum(endprobs(min(nowtimeind+1,numel(endprobs)):end));
prob_start_before_now = sum(startprobs(1:nowtimeind));
costs_now = (t-t(lastrmind)).^2 * prob_end_after_now * prob_start_before_now;

u = startprobs(nowtimeind:end);
v = [t(1), t(1:end-1)].^2;
costsfull = conv(u,v,'full');
costs_late = t*0;
costs_late(nowtimeind:end) = costsfull(1:numel(t)-nowtimeind+1);

A = 4.0;
B = 10;
u = [t(1), t(1:end-1)];
v = -A * startprobs(end:-1:1) ./ (((t(end:-1:1) - t(nowtimeind))/B).^2 + 1);
costsfull = conv(u,v,'full');
costs_early = costsfull(numel(t):-1:1);

costs = binprob * (costs_early + costs_late + costs_now);
costs(1:nowtimeind) = 0;

% cost thresholding:
% costs(costs>80) = 80+0.3*sqrt(costs(costs>80)-80);