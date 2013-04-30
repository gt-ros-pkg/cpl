function [ output_args ] = imagesc3d(img)
%IMAGESC3D Summary of this function goes here
%   Detailed explanation goes here

    [h w t] = size(img);
    imagesc(reshape(img, h, w * t));

end

