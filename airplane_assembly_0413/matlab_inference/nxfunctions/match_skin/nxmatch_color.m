function diff = match_color(img, color)
%MATCH_COLOR Summary of this function goes here
%   Detailed explanation goes here

assert(size(img, 3) == 3 && size(color, 3) == 3);

cc1 = color(1:end/3);
cc2 = color(end/3+1:end/3*2);
cc3 = color(end/3*2+1:end);
id = find(cc1 > 0.005 | cc2 > 0.005 | cc3 > 0.005);

[h w dummy] = size(img);

cimg = cat(3, median(cc1(id)) * ones(h, w) , median(cc2(id)) * ones(h, w) , median(cc3(id)) * ones(h, w) );

diff = img - cimg;
diff = sqrt(diff(:,:,1) .^ 2 + diff(:,:,2) .^ 2 + diff(:,:,3) .^ 2);
end

