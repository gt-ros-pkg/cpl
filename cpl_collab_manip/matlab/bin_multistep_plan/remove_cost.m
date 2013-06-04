function [cost] = remove_cost(t, tr, startprobs, endprobs, binprob, undodur)

if tr > numel(t)
    cost = 1e10*(1+tr-numel(t));
else

    % plateau model:
    % costbeforestart = sum((max(undodur-(t(tr+1:end)-t(tr)),0)).^2 .* startprobs(tr+1:end));
    % costafterstart = sum((undodur).^2.*startprobs(1:tr));
    % costifendafter = costafterstart + costbeforestart;
    % cost = binprob * costifendafter * sum(endprobs(tr+1:end));

    % flat top model:
    % costbeforestart = sum((undodur).^2 .* startprobs(tr+1:end));
    % costafterstart = sum((undodur).^2.*startprobs(1:tr));
    % costifendafter = costafterstart + costbeforestart;
    % cost = binprob * costifendafter * sum(endprobs(tr+1:end));

    % monotonic model:
    m = -0.001;
    % m = 0.0;
    cost = binprob * sum((m*(tr-t(tr+1:end))+undodur).^2 .* endprobs(tr+1:end));

end
