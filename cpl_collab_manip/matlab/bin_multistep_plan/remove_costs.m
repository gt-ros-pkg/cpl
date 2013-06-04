function [costs] = remove_costs(t, startprobs, endprobs, undodur)

for i = 1:numel(t)
    costafterstart = sum((undodur).^2.*startprobs(1:i));
    %lininteg = sum(-(dr+dd).^2.*endprobs(i+1:round(i+(dr+dd)*rate),j)');
    costbeforestart = sum((max(undodur-(t(i+1:end)-t(i)),0)).^2 .* startprobs(i+1:end));
    costifendafter = costafterstart + costbeforestart;
    costs(i) = costifendafter * sum(endprobs(i+1:end));
end
