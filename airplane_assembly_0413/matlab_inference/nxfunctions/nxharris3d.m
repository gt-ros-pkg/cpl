function [px py pt pr r] = nxharris3d(frames, spacevar, timevar, threshold)
%NXHARRIS3D harris spacetime interest point detector
%   frames is h x w x nframe matrix (grayscale image sequence)

%% check inputs
assert(nargin >= 1);

if nargin < 4
    threshold = 0.1;
end
if nargin < 3
    timevar = 8;
end
if nargin < 2
    spacevar = 8;
end

s = 0.5;

%% normalize frames

f = frames;
[h w nframe] = size(f);

f = f - min(f(:));
f = f / max(f(:));
f = f + rand(h, w, nframe) * 1e-5;

%% create gradient

g3 = gk_3d(15, 15, 15, spacevar, timevar);
fg = imfilter(f, g3, 'symmetric');

Ix = imfilter(fg, [-1 0 1], 'symmetric');
Iy = imfilter(fg, [-1; 0; 1], 'symmetric');
dummy(1,1,1) = -1; dummy(1,1,2) = 0; dummy(1,1,3) = 1; 
It = imfilter(fg, dummy, 'symmetric');


%% calculate r

gw = gk_3d(15, 15, 15, s * spacevar, s * timevar);

Ix2 = imfilter(Ix .* Ix, gw, 'symmetric');
Iy2 = imfilter(Iy .* Iy, gw, 'symmetric');
It2 = imfilter(It .* It, gw, 'symmetric');
Ixy = imfilter(Ix .* Iy, gw, 'symmetric');
Iyt = imfilter(Iy .* It, gw, 'symmetric');
Ixt = imfilter(Ix .* It, gw, 'symmetric');

% matrix Ix2 Ixy Ixt
%        Ixy Iy2 Iyt
%        Ixt Iyt It2

k = 0.00005;

r = Ix2 .* Iy2 .* It2 + Ixy .* Iyt .* Ixt + Ixt .* Ixy .* Iyt - ...
    Ixt .* Iy2 .* Ixt - Ixy .* Ixy .* It2 - Ix2 .* Iyt .* Iyt - ...
    k * (Ix2 + Iy2 + It2) .^ 3;

% normalize r
r = r - min(r(:));
r = r / max(r(:));


%% find local maxima, very inefficient implementation
px = [];
py = [];
pt = [];
pr = [];

for x=1+3:w-3
    for y=1+3:h-3
    for t=1+3:nframe-3
    if r(y, x, t) > threshold
        
        k = r(y-2:y+2, x-2:x+2, t-2:t+2);
        check = r(y, x, t) > k;

        if sum(check(:)) == length(check(:)) - 1
            px = [px; x];
            py = [py; y];
            pt = [pt; t];
            pr = [pr; r(y, x, t)];
        end
        
    end
    end
    end
end

[dummy id] = sort(pr, 'descend');
px = px(id);
py = py(id);
pt = pt(id);
pr = dummy;

end

