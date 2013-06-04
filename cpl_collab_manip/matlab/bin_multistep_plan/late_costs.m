function [costs] = late_costs(t, startprobs, endprobs, nowtime)

for i = 1:numel(t)
    if i >= nowtime+1
        %costnow = sum(endprobs(nowtime+1:end)) * sum((t(i)-t(1:nowtime)).^2 .* startprobs(1:nowtime));
        costnow = sum(endprobs(nowtime+1:end)) * sum((t(i)-t(nowtime)).^2 .* startprobs(1:nowtime));
        %costnow = 0;
        costlater = sum((t(i)-t(nowtime+1:i)).^2 .* startprobs(nowtime+1:i));
        costs(i) = costnow + costlater;
    else
        costs(i) = 0;
    end
end
