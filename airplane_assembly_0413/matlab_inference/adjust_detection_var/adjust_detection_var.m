
%
% For noisy: use 25, 1
% For not noisy: use 12, 0.0001
%
if 0
    NAM_NOISE_MODEL = 0
    NAM_NOISY = 0
    KPH_NOISY = -1
end

clc;
learnt_var = m.detection.onedetector.learnt.var

if NAM_NOISE_MODEL
    disp '----------------- adjust detection var'
    disp 'old params'
    m.detection.params.detector_var_scale
    m.detection.onedetector.mean_detection_score
    %% default value
    % m.detection.params.detector_var_scale = 16;
    % m.detection.params.mean_detection_score_factor = 0.02;

    % new value
    % m.detection.params.detector_var_scale           = 21;
    % m.detection.onedetector.mean_detection_score    = 0.2;
    % m.detection.params.detector_var_scale           = 25;
    % m.detection.onedetector.mean_detection_score    = 1;
    % m.detection.params.detector_var_scale           = 12;
    % m.detection.onedetector.mean_detection_score    = 0.01;

    if NAM_NOISY
        m.detection.params.detector_var_prior           = 25;
        m.detection.onedetector.mean_detection_score    = 1;
    else
        m.detection.params.detector_var_prior           = 12;
        m.detection.onedetector.mean_detection_score    = 0.0001;
    end
else

    if KPH_NOISY == 1
        % High noise
        m.detection.params.detector_var_prior           = 0.02 * eye(3);
        m.detection.params.latent_noise                 = 3;
        m.detection.params.future_weight                = 3;
    elseif KPH_NOISY == 0
        % Low noise
        m.detection.params.detector_var_prior           = 0.005 * eye(3);
        m.detection.params.latent_noise                 = 0.1;
        m.detection.params.future_weight                = 0.1;
    else
        % Really low noise
        m.detection.params.detector_var_prior           = 0.005 * eye(3);
        m.detection.params.latent_noise                 = 0.01;
        m.detection.params.future_weight                = 0.01;
    end
end

return;





