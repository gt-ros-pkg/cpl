function [mask label] = filter_mask_by_seed(mask, seed)
%FILTER_MASK_BY_SEED Summary of this function goes here
%   Detailed explanation goes here

    [label n] = bwlabel(mask);
    survivorlabels = unique(label(label > 0 & seed > 0));
    
    mask(:) = 0;
    
    for i=1:n
        if any(i == survivorlabels)
            mask(label == i) = 1;
        else
        end
    end
    
end

