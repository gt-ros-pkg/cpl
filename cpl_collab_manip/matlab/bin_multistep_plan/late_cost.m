function [cost] = late_cost(t, td, startprobs, endprobs, binprob, nowtimeind)

if td > numel(t)
    cost = 1e10;
else
    if td >= nowtimeind+1
        prob_end_after_now = sum(endprobs(nowtimeind+1:end));
        prob_start_before_now = sum(startprobs(1:nowtimeind));
        m = 0.1;
        costbefore = sum(-m .* startprobs(td:end));
        costnow = (t(td)-t(nowtimeind))^2 * prob_end_after_now * prob_start_before_now;
        costlater = sum((t(td)-t(nowtimeind+1:td)).^2 .* startprobs(nowtimeind+1:td));
        cost = binprob*(costbefore + costnow + costlater);
    else
        cost = 0;
    end
end
