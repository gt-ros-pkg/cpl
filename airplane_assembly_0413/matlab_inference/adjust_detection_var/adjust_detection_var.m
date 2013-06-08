
clc;
disp '----------------- adjust detection var'
disp 'old params'
m.detection.onedetector.learnt.var
m.detection.params.detector_var_scale
m.detection.onedetector.mean_detection_score

%% default value
m.detection.params.detector_var_scale = 16;
m.detection.params.mean_detection_score_factor = 0.02;

% new value
m.detection.params.detector_var_scale = 100;
m.detection.params.mean_detection_score_factor = 1;

load training_data

%% update mean detection score

dr = [];

for i=1:size(onedetector.data, 2)
    
    dr(end+1) = mvnpdf(onedetector.data(:,i), onedetector.learnt.mean, m.detection.params.detector_var_scale * onedetector.learnt.var);
    
    if dr(end) > 1
        dr = dr(1:end-1);
    end
end

m.detection.onedetector.mean_detection_score = mean(dr) * m.detection.params.mean_detection_score_factor;

%%
disp 'new params'
m.detection.onedetector.learnt.var
m.detection.params.detector_var_scale
m.detection.onedetector.mean_detection_score





