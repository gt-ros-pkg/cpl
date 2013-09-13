function [] = detector_viz(humacts, slots, binavail, availslot, samp_interval, samp_num, likelihood_params, detector_off)

mintime = samp_interval(1);
nowtime = samp_interval(2);
maxtime = samp_interval(3);
nowtimeind = floor((samp_num-1)*(nowtime-mintime)/(maxtime-mintime))+1;

detect_dists = sample_detector_dists(humacts, slots, binavail, availslot, samp_interval, ...
                                     samp_num, detector_off);
detections = likelihood_function(detect_dists, nowtimeind, likelihood_params);
max_detect = max(detections(:));

figure(7)
clf
t = linspace(mintime, maxtime, samp_num);
for i = 1:numel(binavail)
    subplot(numel(binavail),1,i)
    hold on
    plot(t, detect_dists(i,:))
end

figure(8)
clf
t = linspace(mintime, maxtime, samp_num);
for i = 1:numel(binavail)
    subplot(numel(binavail),1,i)
    xlim([mintime maxtime])
    ylim([0 max_detect])
    hold on
    plot(t, detections(i,:))
end
