function er = ef2(x, cache_e)
%EF Summary of this function goes here
%   Detailed explanation goes here

T = size(x, 2);

if ~exist('cache_e')
    cache_e = zeros(2*T + 1, 1);
    for i=1:2*T + 1
        cache_e(i) = e(i-T-1);
    end
end

er = conv(x, cache_e);
er = er(T+1:2*T);

assert(nxis_convex(er) == 1);

end

