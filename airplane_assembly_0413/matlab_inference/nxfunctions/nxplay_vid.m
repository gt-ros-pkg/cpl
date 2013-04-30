function [ output_args ] = nxplay_vid(vid_volume, ipoints)
%NXPLAY_VID Summary of this function goes here
%   Detailed explanation goes here

if nargin == 1
    ipoints = zeros(0, 3);
end;

xs = ipoints(:,2);
ys = ipoints(:,1);
ts = ipoints(:,3);


for t=1:size(vid_volume, 3)
   
    id = find(ts == t);
    
    imshow(vid_volume(:,:,t));
    hold on;
    plot(xs(id), ys(id), 'r*');
    hold off;
    
    pause(0.1);
    
end

end

