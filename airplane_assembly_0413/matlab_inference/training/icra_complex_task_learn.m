
clear; clc;

model.path = '../icra_complex_task/';
% in this path, load grammar.txt and merged_framesinfo_labels.mat
% save model.mat

%% load grammar
grammar = load_grammar([model.path 'grammar.txt']);
disp 'loading grammar finished'
disp 'press enter'
pause; clc;



%% learn detectors
detection = load('../s/model.mat');
detection = detection.model.detection;

%% remove training & save to data


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






