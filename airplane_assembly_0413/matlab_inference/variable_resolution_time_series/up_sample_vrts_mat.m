function vmat = up_sample_vrts_mat( vmatr, r )
%UP_SAMPLE_VRTS_MAT Summary of this function goes here
%   Detailed explanation goes here

    T0 = sum(r);
    T1 = length(r);
    
    vmat = nan(T0, T0);
    
    
    a = 1;
    for i=1:T1
        
        b = 1;
        for j=1:T1
            
            vmat(a:a-1+r(i),b:b-1+r(j)) = vmatr(i,j) * r(i);
            
            b = b + r(j);
        end
        
        a = a + r(i);
end

