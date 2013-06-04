function [cost] = late_cost(t, td, startprobs, endprobs, binprob, nowtimeind)

if td > numel(t)
    cost = 1e10;
else
    if td >= nowtimeind+1
        costnow = sum(endprobs(nowtimeind+1:end)) * ...
                  sum((t(td)-t(nowtimeind)).^2 .* startprobs(1:nowtimeind));
        costlater = sum((t(td)-t(nowtimeind+1:td)).^2 .* startprobs(nowtimeind+1:td));
        cost = binprob*(costnow + costlater);
    else
        cost = 0;
    end
end
