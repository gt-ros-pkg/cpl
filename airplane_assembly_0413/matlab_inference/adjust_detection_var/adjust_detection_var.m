

if 0
    NAM_NOISE_MODEL = 0
    NAM_NOISY = 0
    KPH_NOISY = -1
end

clc;
learnt_var = m.detection.onedetector.learnt.var

if NAM_NOISE_MODEL
    
    if NAM_NOISY
        m.detection.params.detector_var_scale           = 16; % raw peak is about 120
        m.detection.onedetector.mean_detection_score    = 20;  
        m.detection.onedetector.nam_uniform_component   = 5;
        
    else % standard settings
        m.detection.params.detector_var_scale           = 2; % raw peak is usually 1000
        m.detection.onedetector.mean_detection_score    = 70;
        m.detection.onedetector.nam_uniform_component   = 5;
    end
else

    if KPH_NOISY == 1
        % High noise
        m.detection.params.detector_std_prior  = 0.02;
        m.detection.params.latent_noise        = 3;
        m.detection.params.future_weight       = 3;
    elseif KPH_NOISY == 0
        % Medium noise
        m.detection.params.detector_std_prior  = 0.04;
        m.detection.params.latent_noise        = 0.0;
        m.detection.params.future_weight       = 1.0;
        m.detection.params.pure_detect_weight = 5.0;
    else
        % Low noise
        m.detection.params.detector_std_prior  = 0.04;
        m.detection.params.latent_noise        = 0.75;
        m.detection.params.future_weight       = 1.0;
        m.detection.params.pure_detect_weight = 1.0;
    end
end

return;





