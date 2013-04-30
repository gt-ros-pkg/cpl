function featurevalue = nxdescriptor3d(f, point, volumesize, cubidgrid)
%NXDESCRIPTOR3D Summary of this function goes here
%   Input:  f: h * w * nframe
%           point = [x y t]
%           volumesize = [dx dy dt] should be within h, w, nframe
%           cubidgrid  = [nx ny nt]

[h w nframe] = size(f);

x = point(1);
y = point(2);
t = point(3);

dx = volumesize(1);
dy = volumesize(2);
dt = volumesize(3);

nx = cubidgrid(1);
ny = cubidgrid(2);
nt = cubidgrid(3);

cubidnum = nx * ny * nt;

% calculate gradient & flow
fox = 0;
foy = 0;
fgx = imfilter(f, [-1 0 1], 'symmetric') ;%;+ 1e-10;
fgy = imfilter(f, [-1; 0; 1], 'symmetric'); % + 1e-10;
fgt = 0;

% orientation binning
orientationBinNum = 9;
fgmagnitude = fgx .^ 2 + fgy .^ 2;
bigmagnitudeindex = fgmagnitude > 0.01;

forientationbin = zeros(h, w, nframe); % zero is smal magintude gradient
forientationbin(bigmagnitudeindex) = atan2(fgy(bigmagnitudeindex), fgx(bigmagnitudeindex)) + pi + 1.23; % would be in [0 2pi]+1.23
forientationbin(forientationbin >= 2*pi) = forientationbin(forientationbin >= 2*pi) - 2*pi; % would be in [0 2pi]
forientationbin(forientationbin >= pi) = forientationbin(forientationbin >= pi) - pi; % ignore sign, would be in [0 pi]
%forientationbin(pi - forientationbin < 0.000001) = 0.000001;
%forientationbin(forientationbin < 0.000001) = 0.000001; % would be in (0 pi)
forientationbin(bigmagnitudeindex) = ceil(forientationbin(bigmagnitudeindex) / pi * orientationBinNum ); % would be 1, 2... 9

% 
% forientationbin_mark = zeros(h, w, nframe, orientationBinNum);
% for i=1:orientationBinNum
%     forientationbin_mark(:,:,:,i) = forientationbin == i;
% end


% x1:x2 y1:y2 z1:z2 of volume and cubids

volume = [round(x - dx / 2); round(y - dy / 2); round(t - dt / 2)];
volume(4) = volume(1) + dx - 1;
volume(5) = volume(2) + dy - 1;
volume(6) = volume(3) + dt - 1;

[cx cy ct] = meshgrid(volume(1):(dx / nx):volume(4), volume(2):(dy / ny):volume(5), volume(3):(dt / nt):volume(6));

cubids = [cx(:)'; cy(:)'; ct(:)'];
cubids(4,:) = cubids(1,:) + dx / ny - 1;
cubids(5,:) = cubids(2,:) + dy / ny - 1;
cubids(6,:) = cubids(3,:) + dt / nt - 1;
cubids = round(cubids);

cubids(1,:) = max(1, min(w, cubids(1,:)));
cubids(4,:) = max(1, min(w, cubids(4,:)));
cubids(2,:) = max(1, min(h, cubids(2,:)));
cubids(5,:) = max(1, min(h, cubids(5,:)));
cubids(3,:) = max(1, min(nframe, cubids(3,:)));
cubids(6,:) = max(1, min(nframe, cubids(6,:)));


% gogo
featurevalue = zeros(orientationBinNum, cubidnum);

% compute hog & hof for each cubid
for i=1:cubidnum
    c = cubids(:,i);
    hog = zeros(orientationBinNum, 1);
    for b=1:orientationBinNum
       hog(b) = numel( find( forientationbin(c(2):c(5), c(1):c(4), c(3):c(6)) == b)) + 1e-10;
    end
    % hog = hog / sum(hog(:));
    featurevalue(:,i) = hog;
end

featurevalue = featurevalue(:);

end

