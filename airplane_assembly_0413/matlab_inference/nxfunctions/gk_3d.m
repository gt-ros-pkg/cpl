function g3 = gk_3d(h, w, n, spacevar, timevar)
%GK_3D Summary of this function goes here
%   Detailed explanation goes here

    % center
    cx = round(w / 2);
    cy = round(h / 2);
    ct = round(n / 2);

    %
    
    g3 = zeros(h, w, n);
    
    for x=1:w
        for y=1:h
        for t=1:n

            dx = x - cx;
            dy = y - cy;
            dt = t - ct;

            g = exp(-(dx ^ 2 + dy ^ 2) / (2 * spacevar) - (dt ^ 2) / (2 * timevar));
            g3(y, x, t) = g / sqrt((2 * pi) ^ 3  * spacevar ^ 2 * timevar);

        end
        end
    end
    
    g3 = g3 / sum(g3(:));
    
end

