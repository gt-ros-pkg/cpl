function [px py pr responsemap] = nxharris( img )
%NXHARRIS Harris corner detector
%   Input: double gray image
%   Output: x, y, response of corner points


[h w dummy] = size(img);

% normalize imag
img = img - min(img(:));
img = img / max(img(:));
img = img + rand(h, w) * 1e-5;

% gradient
sigma = 1;
dgx = imfilter(fspecial('gaussian', 15, sigma), [1 0 -1], 'symmetric');
dgy = imfilter(fspecial('gaussian', 15, sigma), [1; 0; -1], 'symmetric');

%dgx = [-1 0 1; -1 0 1; -1 0 1];
%dgy = [-1 -1 -1; 0 0 0; 1 1 1];

imgx = imfilter(img, dgx, 'symmetric');
imgy = imfilter(img, dgy, 'symmetric');

% something something
wsigma = 2;
d = 7;
k = 0.04;
g = fspecial('gaussian', 15, wsigma);

r = zeros([h w]);

imgx2 = imfilter(imgx .* imgx, g, 'symmetric');
imgy2 = imfilter(imgy .* imgy, g, 'symmetric');
img2  = imfilter(imgx .* imgy, g, 'symmetric');

r = (imgx2 .* imgy2 - img2 .* img2) - k .* (((imgx2 + imgy2) / 2) .^ 2);
%r = (imgx2 .* imgy2 - img2 .* img2) ./ (imgx2 + imgy2 + eps); % My preferred  measure.

% normalize & threshold
r = r - min(r(:));
r = r / max(r(:));
r(r < 0.1) = 0;

% non max suppress
px = zeros(h * w, 1); 
py = zeros(h * w, 1);
pr = zeros(h * w, 1);
pcount = 0;
for y=2:h-1
    for x=2:w-1
      
        if r(y, x) > r(y-1, x) && r(y, x) > r(y+1, x) && r(y, x) > r(y, x-1) && r(y, x) > r(y, x+1)
            pcount = pcount + 1;
            px(pcount) = x;
            py(pcount) = y;
            pr(pcount) = r(y, x);
        end
        
    end
end

if pcount == 0
    px = zeros(0, 1); 
    py = zeros(0, 1);
    pr = zeros(0, 1);
else
    px = px(1:pcount);
    py = py(1:pcount);
    pr = pr(1:pcount);
    
    % sort
    [pr id] = sort(pr, 'descend');
    px = px(id);
    py = py(id);
end

responsemap = r;


end

