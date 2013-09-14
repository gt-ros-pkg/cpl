

if 0
    NAM_NOISE_MODEL = 0
    NAM_NOISY = 0
    KPH_NOISY = -1
end

clc;
learnt_var = m.detection.onedetector.learnt.var

if NAM_NOISE_MODEL
    
    if NAM_NOISY == -999 % for test
        m.detection.params.detector_var_scale           = 1; 
        m.detection.onedetector.mean_detection_score    = 1;  
        m.detection.onedetector.nam_uniform_component   = 0;
        
    
    elseif NAM_NOISY == 1
        m.detection.params.detector_var_scale           = 16; % raw peak is about 120
        m.detection.onedetector.mean_detection_score    = 20;  
        m.detection.onedetector.nam_uniform_component   = 5;
        
    elseif NAM_NOISY == 2
        m.detection.params.detector_var_scale           = 25; % raw peak is about 60
        m.detection.onedetector.mean_detection_score    = 20;  
        m.detection.onedetector.nam_uniform_component   = 5;
        
    elseif NAM_NOISY == 3 % standard settings
        m.detection.params.detector_var_scale           = 2; % raw peak is usually 1000
        m.detection.onedetector.mean_detection_score    = 70;
        m.detection.onedetector.nam_uniform_component   = 5;
        
    elseif NAM_NOISY == 143569 % real exp, new kinect setting
        m.detection.params.detector_var_scale           = 4; % raw peak is usually 100-400
        m.detection.onedetector.mean_detection_score    = 2;
        m.detection.onedetector.nam_uniform_component   = 0.01;
        
        
           
        for i=1:length(m.detection.detectors)
           
            m.detection.detectors(i).x_setting.mean          = m.detection.onedetector.learnt.mean;
            m.detection.detectors(i).x_setting.var           = m.detection.onedetector.learnt.var * m.detection.params.detector_var_scale;
            m.detection.detectors(i).x_setting.uni_component = m.detection.onedetector.nam_uniform_component;
            m.detection.detectors(i).x_setting.mean_score    = m.detection.onedetector.mean_detection_score ;
            
        end
        
    elseif NAM_NOISY == 1119 % real exp, new kinect setting, low confidence case
        m.detection.params.detector_var_scale           = 9; % raw peak is usually 150
        m.detection.onedetector.mean_detection_score    = 3;
        m.detection.onedetector.nam_uniform_component   = 0.2;
        
    elseif NAM_NOISY == 1679 % real exp, new kinect setting, low low confidence case
        m.detection.params.detector_var_scale           = 9; % raw peak is usually 150
        m.detection.onedetector.mean_detection_score    = 20;
        m.detection.onedetector.nam_uniform_component   = 10;
        
    elseif NAM_NOISY == 1559 % real exp, new kinect setting, low confidence, observation for bin 12, iros
        
        m.detection.params.detector_var_scale           = 9; % raw peak is usually 150
        m.detection.onedetector.mean_detection_score    = 3;
        m.detection.onedetector.nam_uniform_component   = 0.1;
        
           
        for i=1:length(m.detection.detectors)
           
            m.detection.detectors(i).x_setting.mean          = m.detection.onedetector.learnt.mean;
            m.detection.detectors(i).x_setting.var           = m.detection.onedetector.learnt.var * m.detection.params.detector_var_scale;
            m.detection.detectors(i).x_setting.uni_component = m.detection.onedetector.nam_uniform_component;
            m.detection.detectors(i).x_setting.mean_score    = m.detection.onedetector.mean_detection_score ;
            
        end
        
            m.detection.detectors(11).x_setting.uni_component = 10;
            m.detection.detectors(11).x_setting.mean_score    = 80;
           
    elseif NAM_NOISY == 19 % low confidence for mis calibration
        
        m.detection.params.detector_var_scale           = 9; % raw peak is usually ???
        m.detection.onedetector.mean_detection_score    = 0.5;
        m.detection.onedetector.nam_uniform_component   = 0.05;
           
        for i=1:length(m.detection.detectors)
           
            m.detection.detectors(i).x_setting.mean          = m.detection.onedetector.learnt.mean;
            m.detection.detectors(i).x_setting.var           = m.detection.onedetector.learnt.var * m.detection.params.detector_var_scale;
            m.detection.detectors(i).x_setting.uni_component = m.detection.onedetector.nam_uniform_component;
            m.detection.detectors(i).x_setting.mean_score    = m.detection.onedetector.mean_detection_score ;
            
        end
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
    elseif KPH_NOISY == -1
        % Low noise
        m.detection.params.detector_std_prior  = 0.04;
        m.detection.params.latent_noise        = 0.75;
        m.detection.params.future_weight       = 1.0;
        m.detection.params.pure_detect_weight = 1.0;
    elseif KPH_NOISY == 2
        % IROS high confidence:
        m.detection.params.detector_std_prior  = 0.06;
        m.detection.params.latent_noise        = 0.005;
        m.detection.params.future_weight       = 0.01;
        m.detection.params.pure_detect_weight = 1.0;
    elseif KPH_NOISY == 3
        % IROS low confidence
        m.detection.params.detector_std_prior  = 0.10;
        m.detection.params.latent_noise        = 0.01;
        m.detection.params.future_weight       = 0.05;
        m.detection.params.pure_detect_weight = 1.0;
    elseif KPH_NOISY == 4
        % IROS human high confidence:
        m.detection.params.detector_std_prior  = 0.10;
        m.detection.params.latent_noise        = 0.005;
        m.detection.params.future_weight       = 0.01;
        m.detection.params.pure_detect_weight = 1.0;
    elseif KPH_NOISY == 5
        % IROS human low confidence:
        m.detection.params.detector_std_prior  = 0.15;
        m.detection.params.latent_noise        = 0.005;
        m.detection.params.future_weight       = 0.03;
        m.detection.params.pure_detect_weight = 1.0;
    end
end

return;





