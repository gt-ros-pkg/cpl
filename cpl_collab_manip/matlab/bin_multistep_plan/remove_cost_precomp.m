function [costs] = remove_cost_precomp(t, endprobs, binprob, undo_dur, is_delivered)

m = -0.01;
u = (m*(t-t(end))+undo_dur).^2;
costs = binprob * conv(u, endprobs, 'full');
costs = costs(numel(t):end);

if is_delivered
    A = 0.1;
    prob_rm_after_end = cumsum(endprobs);
    cost_preempt_rm = A*(1./prob_rm_after_end - 1);
    costs = costs + cost_preempt_rm;
end

% if 1
%     dcosts = diff(costs);
%     ddcosts = diff(dcosts);
%     ddcosts(ddcosts<0) = 0;
%     ndcosts = cumsum([dcosts(1), ddcosts]);
%     costs = cumsum([costs(1), ndcosts]);
% end
    
