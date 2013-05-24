function [ vmatr ] = down_sample_vrts_mat( vmat, r)
%DOWN_SAMPLE_VRTS_MAT Summary of this function goes here
%   Detailed explanation goes here
    
    % todo assert
    
    
    %
    T   = size(vmat,1);
    T2  = length(r);
    
    vmatr = 9999 * ones(T2, T2);
    
    a = 1;
    for i=1:T2,
        b = 1;
        for j=1:T2,
            vmatr(i,j) = sum(sum(vmat(a:a-1+r(i),b:b-1+r(j)))) / r(i);
            b = b + r(j);
        end
        a = a + r(i);
    end

end

