function [bin_relevances] = get_bin_relevances(t, probs, slot_states, ...
                                               nowtimeind, endedweight, notbranchweight)

numbins = size(probs,1);
bin_relevances = zeros(1,numbins);
for i = 1:numbins
    binprob = sum(probs{i,1});
    startprobs = probs{i,1} / binprob;
    endprobs = probs{i,2} / binprob;
    if any(i == slot_states)
        bin_relevances(i) = -inf;
    else
        bin_relevances(i) = relevance_heur(t, startprobs, endprobs, binprob, nowtimeind, endedweight, notbranchweight);
    end
end
