function regions = nxsort_regions(regions)
%NXSORT_REGIONS Summary of this function goes here
%   Detailed explanation goes here


    for i=1:length(regions)-1
        for j=i+1:length(regions)
            
            if regions(i).Area > regions(j).Area
               r = regions(i);
               regions(i) = regions(j);
               regions(j) = r;
            end
            
        end;
    end
end

