
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

%% default value
% m.detection.params.detector_var_scale = 16;
% m.detection.params.mean_detection_score_factor = 0.02;

% new value
m.detection.params.detector_var_scale           = 25;
m.detection.onedetector.mean_detection_score    = 1;
m.detection.params.detector_var_scale           = 21;
m.detection.onedetector.mean_detection_score    = 0.2;
m.detection.params.detector_var_scale           = 12;
m.detection.onedetector.mean_detection_score    = 0.001;


return;
% 
% load training_data
% 
% %% update mean detection score
% 
% dr = [];
% 
% for i=1:size(onedetector.data, 2)
%     
%     dr(end+1) = mvnpdf(onedetector.data(:,i), onedetector.learnt.mean, m.detection.params.detector_var_scale * onedetector.learnt.var);
%     
% end
% 
% m.detection.onedetector.mean_detection_score = mean(dr) * m.detection.params.mean_detection_score_factor;
% 
% %%
% disp 'new params'
% m.detection.onedetector.learnt.var
% m.detection.params.detector_var_scale
% m.detection.onedetector.mean_detection_score





