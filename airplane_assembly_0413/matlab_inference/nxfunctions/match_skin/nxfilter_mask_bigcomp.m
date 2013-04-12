function mask = filter_mask_bigcomp(mask)
%FILTER_MASK_BIGCOMP Summary of this function goes here
%   Detailed explanation goes here

    CC = bwconncomp(mask);
    numPixels = cellfun(@numel,CC.PixelIdxList);
    [~,idx] = max(numPixels);
    mask(:) = 0;
    mask(CC.PixelIdxList{idx}) = 1;
    
end

