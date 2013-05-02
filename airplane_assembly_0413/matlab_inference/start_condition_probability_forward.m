function distribution = start_condition_probability_forward(distribution, start_condition)
%START_CONSTRAINT_FORWARD Summary of this function goes here
%   Detailed explanation goes here

    for i=1:length(distribution)-1
       
        if ~start_condition(i)
            distribution(i+1) = distribution(i+1) + distribution(i);
            distribution(i)   = 0;
        end
        
    end

    if ~start_condition(end),
        distribution(end)   = 0;
    end
end

