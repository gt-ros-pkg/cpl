function c = nx_maxdistribution(a, b)
%NX_MAXDISTRIBUTION c = a + b
%   Given 1D distribution of a, and b,
%   Compute distribution of c, where c = max(a, b)

    assert(size(a,1) == 1);
    assert(size(b,1) == 1);
    
    n            = max(length(a), length(b));
    a(1,end+1:n) = 0;
    b(1,end+1:n) = 0;

    % eff impl
    c = a .* (b * triu(ones(n))) + b .* (a * triu(ones(n))) - a .* b;
    
    % simple impl
    if 0
    for v=1:n
        c2(1,v) = a(v) * sum(b(1:v)) + b(v) * sum(a(1:v)) - a(v) * b(v);
    end
    assert(norm(c-c2) < 10e-10);
    end
    
end

