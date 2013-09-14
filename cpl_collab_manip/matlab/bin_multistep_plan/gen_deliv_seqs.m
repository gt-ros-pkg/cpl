function [plans] = gen_deliv_seqs(bin_relevances, max_beam_depth)

[relev_sorted, relev_sorted_inds] = sort(bin_relevances,2,'descend');
bins_sorted = relev_sorted_inds(relev_sorted > -inf);
num_bins_left = numel(bins_sorted);
beam_depth = min(num_bins_left, max_beam_depth);
% beam_counts = beam_depth:-1:1;
beam_counts = [beam_depth:-1:1, ones(1,num_bins_left-beam_depth)];
plans = zeros(prod(beam_counts), numel(beam_counts));
bin_inds = [];
for beam_iter = 1:prod(beam_counts)
    C = beam_iter-1;
    cancel_iter = 0;
    cur_plan = [];
    for i = 1:numel(beam_counts)
        cur_ind = mod(C,beam_counts(i))+1;
        C = floor(C/beam_counts(i));
        nondrawn = setdiff(1:num_bins_left, cur_plan);
        cur_plan(end+1) = nondrawn(cur_ind);
    end
    plans(beam_iter,:) = bins_sorted(cur_plan);
end
        
