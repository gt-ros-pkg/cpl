
%
% For noisy: use 25, 1
% For not noisy: use 12, 0.0001
%

clc;
disp '----------------- adjust detection var'
disp 'old params'
m.detection.onedetector.learnt.var
m.detection.params.detector_var_scale
m.detection.onedetector.mean_detection_score

%% value

m.detection.params.detector_var_scale           = 4;
m.detection.onedetector.mean_detection_score    = 10;

%m.detection.params.detector_var_prior           = 5;
%m.detection.onedetector.mean_detection_score    = 0.001;

% % Low noise
% m.detection.params.detector_var_prior           = 0.1 * eye(3);
% m.detection.params.latent_noise                 = 0.02;
% 
% % High noise
% m.detection.params.detector_var_prior           = 1 * eye(3);
% m.detection.params.latent_noise                 = 0.2;

return;





