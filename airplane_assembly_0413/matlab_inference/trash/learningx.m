

clc; clear; 
close all;

load s/merged_framesinfo_labels

data.T                  = 900;
data.detector_var_scale = 16;
data.duration_var_scale = 9;
data.use_onedetector    = 1;
data.downsample_ratio   = 7;

grammar = load_grammar('./s/grammar.txt');


detectors   = struct('trainingdata', {});
onedetector = struct('trainingdata', []);

%% associate symbols and bin
%  todo


for i=1:length(grammar.symbols)
    if grammar.symbols(i).is_terminal
        grammar.symbols(i).bin_id = actionname2binid(grammar.symbols(i).name);
    end
end

grammar.symbols(actionname2symbolid('Body', grammar)).bin_id        = 3;
grammar.symbols(actionname2symbolid('Body', grammar)).wait_for_bin  = 1;

grammar.symbols(actionname2symbolid('Nose_A', grammar)).bin_id        = 11;
grammar.symbols(actionname2symbolid('Nose_A', grammar)).wait_for_bin  = 1;

grammar.symbols(actionname2symbolid('Nose_H', grammar)).bin_id        = 10;
grammar.symbols(actionname2symbolid('Nose_H', grammar)).wait_for_bin  = 1;

grammar.symbols(actionname2symbolid('Wing_AT', grammar)).bin_id        = 12;
grammar.symbols(actionname2symbolid('Wing_AT', grammar)).wait_for_bin  = 1;

grammar.symbols(actionname2symbolid('Wing_AD', grammar)).bin_id        = 2;
grammar.symbols(actionname2symbolid('Wing_AD', grammar)).wait_for_bin  = 1;

grammar.symbols(actionname2symbolid('Wing_H', grammar)).bin_id        = 7;
grammar.symbols(actionname2symbolid('Wing_H', grammar)).wait_for_bin  = 1;

grammar.symbols(actionname2symbolid('Tail_AT', grammar)).bin_id        = 14;
grammar.symbols(actionname2symbolid('Tail_AT', grammar)).wait_for_bin  = 1;

grammar.symbols(actionname2symbolid('Tail_AD', grammar)).bin_id        = 15;
grammar.symbols(actionname2symbolid('Tail_AD', grammar)).wait_for_bin  = 1;

grammar.symbols(actionname2symbolid('Tail_H', grammar)).bin_id        = 13;
grammar.symbols(actionname2symbolid('Tail_H', grammar)).wait_for_bin  = 1;



%% train detector for each bin

disp ==========================================================
disp 'train detector for each bin'
disp ==========================================================

for i=1:length(label)
    
    l = label(i);
    
    if strcmp(l.name, 'start') || strcmp(l.name, 'end')
        continue;
    end

    binid = actionname2binid(l.name);
    
    if isempty(frames_info(l.start).bins(binid).H)
        disp(['Too bad, missing marker ' num2str(binid) ' for ' l.name ' at this frame:']);
        disp(frames_info(l.start).s);
        continue;
    end
    
    lh = inv(frames_info(l.start).bins(binid).H) * [frames_info(l.start).lefthand; 1];
    rh = inv(frames_info(l.start).bins(binid).H) * [frames_info(l.start).righthand; 1];
    
    closest_hand = lh(1:3);
    if norm(rh) < norm(lh) || isnan(norm(closest_hand))
        closest_hand = rh(1:3);
    end;
    if isnan(norm(closest_hand))
        disp(['Too bad, missing hands for ' l.name ' at this frame:']);
        disp(frames_info(l.start).s);
        continue;
    end
    
    detectors(binid).exist = 1;
    detectors(binid).trainingdata(:,end+1) = closest_hand;
    onedetector.trainingdata(:,end+1) = closest_hand;
end

for i=1:length(detectors)
   if detectors(i).exist == 1
       
       assert(size(detectors(i).trainingdata, 2) > 1);
       
       detectors(i).learnt.mean = mean(detectors(i).trainingdata')';
       detectors(i).learnt.var  = cov(detectors(i).trainingdata') + eye(3) * 0.0001;
       
        % print
        disp(['Train detector for bin ' num2str(i)]);
        disp data
        disp(detectors(i).trainingdata');
        disp mean
        disp(detectors(i).learnt.mean');
        disp var
        disp(detectors(i).learnt.var);
   end
end

onedetector.learnt.mean = mean(onedetector.trainingdata')';
onedetector.learnt.var  = cov(onedetector.trainingdata') + eye(3) * 0.0001;

%% train average detection score

disp ==========================================================
disp 'train average detection score'
disp ==========================================================

for d=1:length(detectors)
if detectors(d).exist == 1
    
    dr = [];

    for t=1:length(frames_info)

        if isempty(frames_info(t).bins(d).H)
            dr(end+1) = 0;
            continue;
        end
        
        lh = inv(frames_info(t).bins(d).H) * [frames_info(t).lefthand; 1];
        rh = inv(frames_info(t).bins(d).H) * [frames_info(t).righthand; 1];

        
        closest_hand = lh(1:3);
        if norm(rh) < norm(lh) || isnan(norm(closest_hand))
            closest_hand = rh(1:3);
        end;
        if isnan(norm(closest_hand))
            dr(end+1) = 0;
            continue;
        end
        
        if data.use_onedetector
            dr(end+1) = mvnpdf(closest_hand, onedetector.learnt.mean, data.detector_var_scale * onedetector.learnt.var);
        else
            dr(end+1) = mvnpdf(closest_hand, detectors(d).learnt.mean, data.detector_var_scale * detectors(d).learnt.var);
        end
    end

    detectors(d).mean_detection_score = mean(dr) / 50;
    disp(['Mean detection score of bin ' num2str(d) ' is ' num2str(detectors(d).mean_detection_score)]);
    % plot
    figure(d); 
    plot(dr);
    hold on;
    for i=1:length(label)
        if strcmp(label(i).name, 'start') || strcmp(label(i).name, 'end')
            continue;
        end
        if actionname2binid(label(i).name) == d
            plot(label(i).start, dr(label(i).start), '*r');
        end
    end
    hold off;
    xlabel('Time')
    ylabel(['Bin ' num2str(d) ' Detector'])
end
end

onedetector.mean_detection_score = mean([detectors.mean_detection_score]);

%% train action duration

disp ==========================================================
disp 'train action duration'
disp ==========================================================

grammar.symbols(1).duration_data = [];

for i=1:length(label)
    
    l = label(i);
    
    if strcmp(l.name, 'start') || strcmp(l.name, 'end')
        continue;
    end

    symbolid = actionname2symbolid(l.name, grammar);
    
    grammar.symbols(symbolid).duration_data(:,end+1) = l.end - l.start + 1;

end

for i=1:length(grammar.symbols)
    if ~isempty(grammar.symbols(i).duration_data)
        
        grammar.symbols(i).learntparams.duration_mean = mean(grammar.symbols(i).duration_data);
        grammar.symbols(i).learntparams.duration_var  = var(grammar.symbols(i).duration_data);
        
        disp(['Train duration for action ' grammar.symbols(i).name]);
        disp data
        disp(grammar.symbols(i).duration_data');
        disp mean
        disp(grammar.symbols(i).learntparams.duration_mean);
        disp var
        disp(grammar.symbols(i).learntparams.duration_var);
    end
end

%% save
data.grammar     = grammar;
data.detectors   = detectors;
data.onedetector = onedetector;

clearvars -except data;

save s/learntdata

%% construct inference structure m
% 
% clear; clc; load s/learntdata
% 
% m   = struct;
% m.T = data.T;
% m.g = [];
% 
% x   = [1];
% xm  = [1];
% i   = 0;
% while 1
%     i = i + 1;
%     if length(x) < i, break; end;
%     
%     s = data.grammar.symbols(x(i));
%     r = data.grammar.rules(find([data.grammar.rules.left] == x(i)));
%     
%     m.g(end+1).id           = x(i);
%     m.g(end).isterminal     = s.is_terminal;
%     m.g(end).bin_id         = s.bin_id;
%     m.g(end).wait_for_bin   = s.wait_for_bin;
%     
%     if s.is_terminal
%         
%         duration_mean = data.grammar.symbols(m.g(end).id).learntparams.duration_mean / data.downsample_ratio;
%         duration_var  = data.duration_var_scale * data.grammar.symbols(m.g(end).id).learntparams.duration_var / data.downsample_ratio^2;
%         duration      = nxmakegaussian(m.T, duration_mean, duration_var);
%         durationmat   = zeros(m.T,m.T);
%         
%         for j=1:m.T
%             durationmat(j,j:end) = duration(1:m.T-j+1);
%         end
%         
%         m.g(end).durationmat = durationmat;
%         m.g(end).likelihood = triu(ones(m.T, m.T));
%         m.g(end).log_null_likelihood = log(1);
%         
%         m.g(end).obv_duration_likelihood = m.g(end).likelihood .* m.g(end).durationmat;
%         
%     else
%         
%         m.g(end).prule = r.right; % todo
%         m.g(end).prule = length(x) + [1:length(r.right)];
%         m.g(end).andrule = ~r.or_rule;
%         m.g(end).orweights = r.or_prob;
%         
%         x = [x r.right];
%     end
%     
%     
% end
% 
% m.s = 1;
% m.g(1).start_distribution = 0 * ones(1, data.T) / data.T;
% m.g(1).start_distribution(1:data.T) = 1 / data.T;
% m.g(1).end_likelihood = ones(1, data.T) / data.T;
% 
% clearvars -except data m;
% 
% save s/learntdata































