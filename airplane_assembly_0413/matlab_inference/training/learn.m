
clear; clc;

model.path = '../s/';
% in this path, load grammar.txt and merged_framesinfo_labels.mat
% save model.mat

%% load grammar
grammar = load_grammar([model.path 'grammar.txt']);
disp 'loading grammar finished'
disp 'press enter'
pause; clc;

%%
disp 'loading training data...'
load([model.path 'merged_framesinfo_labels'])
disp 'loading training data finished'
disp 'press enter'
pause; clc;


%% learn duration
learn_duration  
disp 'learning duration finished'
disp 'press enter'
pause; clc;

clearvars -except model grammar frames_info label

%% learn detectors
learn_detectors
disp 'learning detectors finished'
disp 'press enter'
pause; clc; close all;

clearvars -except model grammar detection

%% remove training & save to data

grammar.symbols         = rmfield(grammar.symbols, 'duration_data');
detection.detectors    	= rmfield(detection.detectors, 'trainingdata');
detection.onedetector  	= rmfield(detection.onedetector, 'trainingdata');

model.grammar        = grammar;
model.detection      = detection;


%% save

clearvars -except model
save([model.path 'model'])

if 0
    file = fopen([model.path 'model.xml'], 'wt');
    fwrite(file, nx_toxmlstr(model, 'model'));
    fclose(file);
end






