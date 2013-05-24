function detections = run_action_detections( frame_info, data )
%RUN_ACTION_DETECTIONS Summary of this function goes here
%   Detailed explanation goes here

    detections = nan(1, length(data.detectors));
    
    for d=1:length(data.detectors)
    if data.detectors(d).exist & ~isempty(frame_info.bins(d).H)
        
        lh = inv(frame_info.bins(d).H) * [frame_info.lefthand; 1];
        rh = inv(frame_info.bins(d).H) * [frame_info.righthand; 1];

        % check hand exist
        closest_hand = lh(1:3);
        if norm(rh) < norm(lh) || isnan(norm(closest_hand))
            closest_hand = rh(1:3);
        end;
        if isnan(norm(closest_hand))
            continue;
        end
        
        % run detector
        if data.params.use_onedetector
            detections(d) = mvnpdf(closest_hand, data.onedetector.learnt.mean, data.params.detector_var_scale * data.onedetector.learnt.var) / data.detectors(d).mean_detection_score;
        else
            detections(d) = mvnpdf(closest_hand, data.detectors(d).learnt.mean, data.params.detector_var_scale * data.detectors(d).learnt.var) / data.detectors(d).mean_detection_score;
        end
    end
    end
    
    
end

