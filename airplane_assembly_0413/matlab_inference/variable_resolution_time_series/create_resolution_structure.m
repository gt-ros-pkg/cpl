function rs = create_resolution_structure(T0, center_point, rate)
%CREATE_RESOLUTION_STRUCTURE Summary of this function goes here
%   Detailed explanation goes here


rs          = struct;
rs.csize    = 1;

csize1 = [];
i      = 1;
while sum(csize1) < center_point - 1
    csize1 = [round(1 * rate ^ i) csize1];
    i = i + 1;
end

csize1(1) = 0;
csize1(1) = center_point - 1 - sum(csize1);

csize2 = [];
i      = 1;
while sum(csize2) < T0 - center_point
    csize2 = [csize2 round(1 * rate ^ i)];
    i = i + 1;
end

csize2(end) = 0;
csize2(end) = T0 - center_point - sum(csize2);

rs.csize = [csize1 1 csize2];

% find start point & end point
rs.T0 = T0;
rs.T  = length(rs.csize);

for i=1:rs.T
    if i == 1,
        rs.start(i) = 1;
    else
        rs.start(i) = rs.end(i-1) + 1;
    end
    
    rs.end(i) = rs.start(i) + rs.csize(i) - 1;
end

end

