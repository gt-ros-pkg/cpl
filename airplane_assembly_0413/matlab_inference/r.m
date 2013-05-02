
clc; clear; close all;

load learntdata;
load merged_framesinfo_labels;

detections = [];

%%
for t=1:m.T * data.downsample_ratio
    
    nt =  t / data.downsample_ratio;
    
    % evaluator
    tic;
    dr = run_action_detections(frames_info(t), data);
    dr(isnan(dr)) = 1;
    toc;
    
    % check downsampleratio
    detections(end+1,:) = dr;
    if size(detections,1) < data.downsample_ratio
        continue;
    end
    detections = max(detections, [], 1);
    
    % update dr
    disp(['Frame ' num2str(t)]);
    tic;
    for i=1:length(m.g)
        if m.g(i).isterminal
            actionname  = data.grammar.symbols(m.g(i).id).name;
            d = actionname2binid(actionname);
            m.g(i).obv_duration_likelihood(nt,nt:end) = detections(d) * m.g(i).durationmat(nt,nt:end);
        end
    end
    toc;
    
    detections = [];
    
    % inference
    tic;
    m = m_inference_v2(m, data, 1);
    data.grammar.symbols = calculate_symbol_distribution(m, data.grammar.symbols);
    toc; 
    
    % plot
    cla
    hold on;
    for action = {'Nose_A', 'wing_ad3', 'tail_ad1', 'tail_ad2', 'tail_ad3'}
        plot(data.grammar.symbols(actionname2symbolid(action{1}, data.grammar)).end_distribution, 'color', nxtocolor(sum(action{1})));
    end
    plot(nt, 0, '*black');
    hold off;
    legend({'Nose_A', 'wing_ad3', 'tail_ad1', 'tail_ad2', 'tail_ad3'});
    disp('P');
    disp(sum(data.grammar.symbols(actionname2symbolid('wing_at1', data.grammar)).end_distribution));
    disp(sum(data.grammar.symbols(actionname2symbolid('wing_ad1', data.grammar)).end_distribution));
    disp(sum(data.grammar.symbols(actionname2symbolid('wing_h1', data.grammar)).end_distribution));
    pause(0.01);

end
