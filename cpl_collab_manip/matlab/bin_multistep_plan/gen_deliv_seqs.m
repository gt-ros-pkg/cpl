function [plans] = gen_deliv_seqs(t, beam_counts, probs, bins, slot_states, ...
                                  nowtimeind, endedweight, notbranchweight)

bin_relevances = zeros(1,numel(bins));
for i = 1:numel(bins)
    binprob = sum(probs{i,1});
    startprobs = probs{i,1} / binprob;
    endprobs = probs{i,2} / binprob;
    if any(bins(i) == slot_states)
        bin_relevances(i) = -inf;
    else
        bin_relevances(i) = relevance_heur(t, startprobs, endprobs, binprob, nowtimeind, endedweight, notbranchweight);
    end
end
[relev_sorted, relev_sorted_inds] = sort(bin_relevances,2,'descend');
bins_sorted = bins(relev_sorted_inds);
%bin_relevances
plans = zeros(prod(beam_counts), numel(beam_counts));
bin_inds = [];
for beam_iter = 1:prod(beam_counts)
    C = beam_iter-1;
    cancel_iter = 0;
    cur_plan = [];
    for i = 1:numel(beam_counts)
        cur_ind = mod(C,beam_counts(i))+1;
        C = floor(C/beam_counts(i));
        nondrawn = setdiff(1:numel(bins), cur_plan);
        cur_plan(end+1) = nondrawn(cur_ind);
    end
    plans(beam_iter,:) = bins_sorted(cur_plan);
end
        
