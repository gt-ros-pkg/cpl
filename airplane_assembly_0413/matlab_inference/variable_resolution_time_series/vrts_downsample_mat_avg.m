function downsampled_v = vrts_downsample_mat_avg(v, rs)
%VRTS_DOWNSAMPLE_MAT_AVG Summary of this function goes here
%   Detailed explanation goes here



    [h w] = size(v);
    
    assert(h == rs.T0 && w == rs.T0);

    downsampled_v = nan(rs.T, rs.T);
    
    for i=1:rs.T
        for j=1:rs.T
            downsampled_v(i,j) = sum(sum(v(rs.start(i):rs.end(i),rs.start(j):rs.end(j)))) / rs.csize(i) / rs.csize(j);
        end
    end
    
    
end

