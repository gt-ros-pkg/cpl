

disp ==========================================================
disp 'train detector for each bin'
disp ==========================================================

params.use_onedetector              = 1;
params.detector_var_scale           = 16;
params.mean_detection_score_factor	= 1/50;

detectors   = struct('trainingdata', {});
onedetector = struct('trainingdata', []);

for i=1:length(label)
    
    l = label(i);
    
    if strcmp(l.name, 'start') || strcmp(l.name, 'end')
        continue;
    end

    detector_id = actionname2detectorid(l.name, grammar);
    binid       = detector_id;
    
    if isempty(frames_info(l.start).bins(binid).H)
        disp(['Too bad, missing marker ' num2str(binid) ' for ' l.name ' at this frame:']);
        disp(frames_info(l.start).s);
        continue;
    end
    
    lh = inv(frames_info(l.start).bins(binid).H) * [frames_info(l.start).lefthand; 1];
    rh = inv(frames_info(l.start).bins(binid).H) * [frames_info(l.start).righthand; 1];
    
    closest_hand = lh(1:3);
    if norm(rh) < norm(lh) || isnan(norm(closest_hand))
        closest_hand = rh(1:3);
    end;
    if isnan(norm(closest_hand))
        disp(['Too bad, missing hands for ' l.name ' at this frame:']);
        disp(frames_info(l.start).s);
        continue;
    end
    
    detectors(detector_id).exist = 1;
    detectors(detector_id).trainingdata(:,end+1) = closest_hand;
    onedetector.trainingdata(:,end+1) = closest_hand;
end

for i=1:length(detectors)
   if detectors(i).exist == 1
       
       assert(size(detectors(i).trainingdata, 2) > 1);
       
       detectors(i).learnt.mean = mean(detectors(i).trainingdata')';
       detectors(i).learnt.var  = cov(detectors(i).trainingdata') + eye(3) * 0.0001;
       
        % print
        disp(['Train detector for bin ' num2str(i)]);
        disp data
        disp(detectors(i).trainingdata');
        disp mean
        disp(detectors(i).learnt.mean');
        disp var
        disp(detectors(i).learnt.var);
   end
end

onedetector.learnt.mean = mean(onedetector.trainingdata')';
onedetector.learnt.var  = cov(onedetector.trainingdata') + eye(3) * 0.0001;


%% train average detection score

disp ==========================================================
disp 'train average detection score'
disp ==========================================================

for d=1:length(detectors)
if detectors(d).exist == 1
    
    dr = [];

    for t=1:length(frames_info)

        if isempty(frames_info(t).bins(d).H)
            dr(end+1) = 0;
            continue;
        end
        
        lh = inv(frames_info(t).bins(d).H) * [frames_info(t).lefthand; 1];
        rh = inv(frames_info(t).bins(d).H) * [frames_info(t).righthand; 1];

        
        closest_hand = lh(1:3);
        if norm(rh) < norm(lh) || isnan(norm(closest_hand))
            closest_hand = rh(1:3);
        end;
        if isnan(norm(closest_hand))
            dr(end+1) = 0;
            continue;
        end
        
        if params.use_onedetector
            dr(end+1) = mvnpdf(closest_hand, onedetector.learnt.mean, params.detector_var_scale * onedetector.learnt.var);
        else
            dr(end+1) = mvnpdf(closest_hand, detectors(d).learnt.mean, params.detector_var_scale * detectors(d).learnt.var);
        end
    end

    detectors(d).mean_detection_score = mean(dr) * params.mean_detection_score_factor;
    disp(['Mean detection score of bin ' num2str(d) ' is ' num2str(detectors(d).mean_detection_score)]);
    % plot
    figure(d); 
    plot(dr);
    hold on;
    for i=1:length(label)
        if strcmp(label(i).name, 'start') || strcmp(label(i).name, 'end')
            continue;
        end
        if actionname2detectorid(label(i).name, grammar) == d
            plot(label(i).start, dr(label(i).start), '*r');
        end
    end
    hold off;
    xlabel('Time')
    ylabel(['Bin ' num2str(d) ' Detector'])
end
end

onedetector.mean_detection_score = mean([detectors.mean_detection_score]);
 
detection.detectors   = detectors;
detection.params      = params;
detection.onedetector = onedetector;

