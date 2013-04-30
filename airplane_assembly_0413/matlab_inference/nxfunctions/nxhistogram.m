function hst = nxhistogram(f, binnums, mins, maxs)
%NXHISTOGRAM Summary of this function goes here
%   Detailed explanation goes here

    [h w c] = size(f);
    binnumtotal = 1;
    
    binvalue = zeros(h * w, 1);
    
    for i=1:c
       
       % normalize to [0, 1]
       fc = f(:,:,i);
       fc = double(fc(:));
       fc = (fc - mins(i)) / (maxs(i) - mins(i));
       assert(length(find(fc < 0 | fc > 1)) == 0, 'you have to provide the correct min & max');
       
       % devide by number of bin
       binnum = binnums(i);
       binnumtotal = binnumtotal * binnum;
       fc = floor(fc * binnum); % should be 0, 1, ... binnum
       fc(fc == binnum) = binnum - 1; % should be 0, 1, ... (binnum-1)
       
       binvalue = binvalue * binnum + fc;
       
    end

    % binvalue should have value from 0,1... (binnumtotal-1)
    % now we count
    hst = zeros(binnumtotal, 1);
    for i=0:binnumtotal-1
       hst(i+1) = length(find(binvalue == i)); 
    end
    
end

