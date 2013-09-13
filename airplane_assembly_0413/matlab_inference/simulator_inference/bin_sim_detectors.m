function [detections, dists, min_hand_pos] = bin_sim_detectors(frame_info, data, min_hand_pos)

detections = nan(1, length(data.detectors));
dists = nan(1, length(data.detectors));

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
            detections(d) = (mvnpdf(closest_hand, cur_mean, cur_var) + data.onedetector.nam_uniform_component) / detect_score;
        else
            cur_mean = data.onedetector.learnt.mean;
            cur_std = data.params.detector_std_prior;
            pure_detect_weight = data.params.pure_detect_weight;
            latent_noise = data.params.latent_noise;
            future_weight = data.params.future_weight;
            max_norm_pdf = normpdf(0,0,cur_std);
            
            %new_mean = [0.0837,-0.0772,-0.0935]';
            % new_mean = cur_mean;
            % closest_hand = closest_hand;%-[0.0837,-0.0772,-0.0935]';
            dists(d) = norm(closest_hand-cur_mean);
            if dists(d) < min_hand_pos(1)
                min_hand_pos = [dists(d), closest_hand'];
            end
            hand_dist = norm(closest_hand - cur_mean);
            detected_lik = normpdf(hand_dist, 0, cur_std)/ max_norm_pdf;
            detections(d) = (detected_lik*pure_detect_weight + latent_noise) / future_weight;
            % detections(d) = (mvnpdf(closest_hand, 0*cur_mean, cur_var) + ...
            %                  latent_noise) / future_weight;
            
        end
    else
        cur_mean = data.onedetector.learnt.mean;
        cur_var = data.params.detector_var_scale * data.detectors(d).learnt.var;
        detect_score = data.detectors(d).mean_detection_score;
        detections(d) = mvnpdf(closest_hand, cur_mean, cur_var) / detect_score;
        disp('THIS SHOULD NOT RUN')
    end
end
end
