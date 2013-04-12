function r = integralImage(img)
%INTEG Summary of this function goes here
%   Detailed explanation goes here
    
    r = cumsum(cumsum(double(img)),2);

end

