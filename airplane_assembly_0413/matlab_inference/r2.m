
load s/merged_framesinfo_labels;
m = gen_inference_net('s/model');

detection_raw_result = ones(length(m.detection.result), m.params.T);

for t=1:m.params.T*m.params.downsample_ratio
    
    if mod(t, m.params.downsample_ratio) > 0
        continue;
    end
    nt = t / m.params.downsample_ratio;
    
    d = run_action_detections(frames_info(t), m.detection);
    d(find(isnan(d))) = 1;
    detection_raw_result(:,nt) = d;
    
    for i=1:length(m.detection.result)
        m.detection.result{i} = triu(repmat(detection_raw_result(i,:)', [1 m.params.T]));
    end
    
    
    figure(1);
    imagesc(detection_raw_result);
    
 	m = m_inference_v3(m);
    
    figure(2);
    m_plot_distributions(m, {'Body', 'Nose_A', 'Wing_AT'}, {'S'});
    pause(0.01);
    
end


m = m_inference_v3(m);
m_plot_distributions(m, {'Body', 'Nose_A', 'Wing_AT'}, {'S'});





















