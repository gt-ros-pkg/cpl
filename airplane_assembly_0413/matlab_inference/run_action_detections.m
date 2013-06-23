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
            if data.params.nam_noise_model
                cur_mean = data.onedetector.learnt.mean;
                cur_var = data.params.detector_var_scale * data.onedetector.learnt.var;
                detect_score = data.onedetector.mean_detection_score;
                detections(d) = mvnpdf(closest_hand, cur_mean, cur_var) / detect_score;
            else
                cur_mean = data.onedetector.learnt.mean;
                cur_var = data.params.detector_var_prior + data.onedetector.learnt.var;
                latent_noise = data.params.latent_noise;
                future_weight = data.params.future_weight;
                detections(d) = (mvnpdf(closest_hand, cur_mean, cur_var) + latent_noise) ...
                                 / future_weight;
            end
        else
            cur_mean = data.onedetector.learnt.mean;
            cur_var = data.params.detector_var_scale * data.detectors(d).learnt.var;
            detect_score = data.detectors(d).mean_detection_score;
            detections(d) = mvnpdf(closest_hand, cur_mean, cur_var) / detect_score;
            % disp('THIS SHOULD NOT RUN')
        end
    end
    end
    
    
end

