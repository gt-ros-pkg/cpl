function [ output_args ] = show_segments(im, segments)
%SHOW_SEGMENTS Summary of this function goes here
%   Detailed explanation goes here


    [h w c] = size(im);
    
    for y=2:h-1
        for x=2:w-1
            border = 9 * segments(y,x) ~= sum(sum(segments(y-1:y+1,x-1:x+1)));
            if border
                im(y, x, :) = 0;
            end
        end
    end

    imshow(im);
    
end

