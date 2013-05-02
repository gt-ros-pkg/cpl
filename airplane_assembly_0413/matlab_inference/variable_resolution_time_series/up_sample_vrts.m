function v = up_sample_vrts( vr, r )
%UP_SAMPLE_VRTS Summary of this function goes here
%   Detailed explanation goes here
    
    T  = sum(r);
    T2 = length(r);
    
    assert(length(vr) == length(r));
    
    j = 1;
    for i=1:T2,
        v(j:j+r(i)-1) = vr(i) / r(i);
        j = j + r(i);
    end;

end

