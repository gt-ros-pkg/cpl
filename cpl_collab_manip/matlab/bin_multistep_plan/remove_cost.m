function [cost] = remove_cost(t, tr, startprobs, endprobs, binprob, undo_dur)

if tr > numel(t)
    cost = 1e10*(1+tr-numel(t));
else

    % plateau model:
    % costbeforestart = sum((max(undo_dur-(t(tr+1:end)-t(tr)),0)).^2 .* startprobs(tr+1:end));
    % costafterstart = sum((undo_dur).^2.*startprobs(1:tr));
    % costifendafter = costafterstart + costbeforestart;
    % cost = binprob * costifendafter * sum(endprobs(tr+1:end));

    % flat top model:
    % costbeforestart = sum((undo_dur).^2 .* startprobs(tr+1:end));
    % costafterstart = sum((undo_dur).^2.*startprobs(1:tr));
    % costifendafter = costafterstart + costbeforestart;
    % cost = binprob * costifendafter * sum(endprobs(tr+1:end));

    % monotonic model:
    m = -0.001;
    % m = 0.0;
    cost = binprob * sum((m*(tr-t(tr+1:end))+undo_dur).^2 .* endprobs(tr+1:end));

end
