function r = nxis_convex( vec )
%NXIS_CONVEX Summary of this function goes here
%   Detailed explanation goes here

    n = length(vec);
    
    for i=2:n-1
        
        if vec(i-1) + vec(i+1) < 2 * vec(i)
            r = 0;
            return;
        end
        
    end
    
    r = 1;
    return;
end

