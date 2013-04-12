function v = nxifelse(condition, true_return, false_return)
%NXIFELSE Summary of this function goes here
%   Detailed explanation goes here
    
    if condition
        v = true_return;
    else
        v = false_return;
    end;

end

