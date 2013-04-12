function [a b] = nx_maxdistribution_backward( c )
%NX_MAXDISTRIBUTION_BACKWARD reverse of nx_maxdistribution
%   Given likelihood P(Z | c), compute P(Z | a) and P(Z | b)

    n = length(c);
    
    % simple impl
    for v=1:n
        a(v) = c(v) * (v / n) + (1 / n) * sum(c(v+1:n));
        b(v) = c(v) * (v / n) + (1 / n) * sum(c(v+1:n));
    end
end

