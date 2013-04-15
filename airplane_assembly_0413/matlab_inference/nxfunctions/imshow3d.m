function [ output_args ] = imshow3d(img)
%IMSHOW3D Summary of this function goes here
%   Detailed explanation goes here

    nframe = size(img, 3);
    im = [];
    
    for i=1:nframe
        im = [im img(:,:,i)];
    end

    imshow(im);
    
end

