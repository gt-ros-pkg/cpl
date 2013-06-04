function [costs] = late_costs(t, startprobs, endprobs, dr, dd)

for i = 1:numel(t)
    lateinteg = sum((max(t(i)-t,0)).^2.*startprobs);
    costs(i) = lateinteg .* sum(endprobs(i+1:end));
end
