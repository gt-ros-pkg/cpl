function vr = down_sample_vrts( v, r )
%UP_SAMPLE_VRTS Summary of this function goes here
%   Detailed explanation goes here
    
    T = length(v);
    assert(T == sum(r));
    
    T2 = length(r);
    vr = 9999 * ones(1, T2);
    
    j = 1;
    for i=1:T2
        vr(i) = sum(v(j:j+r(i)-1));
        j = j + r(i);
    end

end

