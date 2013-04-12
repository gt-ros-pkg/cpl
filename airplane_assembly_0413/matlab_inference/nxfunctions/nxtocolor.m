function c = nxtocolor(i)
%NXTOCOLOR Summary of this function goes here
%   Detailed explanation goes here

    colors = [1 0 0; 0 1 0; 0 0 1; 1 1 0; 0 1 1; 1 0 1;
        0 0 0; 0.4 0.4 0.4; 1 0.5 0; 1 0 0.5; 0.5 1 0; 0 1 0.5; 0 0.5 1; 0.5 0 1;
        0.5 0 0; 0 0.5 0; 0 0 0.5; 0.3 1 0.3; 1 0.3 0.3; 0.3 0.3 1];

    c = colors(1 + mod(i,size(colors,1)),:);
end

