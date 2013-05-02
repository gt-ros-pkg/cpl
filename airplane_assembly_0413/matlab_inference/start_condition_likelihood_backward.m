function distribution = start_condition_likelihood_backward(distribution, start_condition)
%START_CONDITION_LIKELIHOOD_BACKWARD Summary of this function goes here
%   Detailed explanation goes here

    if ~start_condition(end)
        distribution(end) = 0;
    end

    for i=length(distribution)-1:-1:1
        if ~start_condition(i),
            distribution(i) = distribution(i+1);
        end
    end



end

