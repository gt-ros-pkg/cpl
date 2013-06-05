function [plans] = gen_deliv_seqs(bin_relevances, beam_counts)

[relev_sorted, relev_sorted_inds] = sort(bin_relevances,2,'descend');
bins_sorted = relev_sorted_inds;
numbins = numel(bin_relevances);
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
        nondrawn = setdiff(1:numbins, cur_plan);
        cur_plan(end+1) = nondrawn(cur_ind);
    end
    plans(beam_iter,:) = bins_sorted(cur_plan);
end
        
