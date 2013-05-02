

%%
clc; clear; close all;

input_label_files   = {'../s/label1.txt', '../s/label2.txt', '../s/label3.txt'};
input_frames_files  = {'../s/frames1.txt', '../s/frames2.txt', '../s/frames3.txt'};
output              = '../s/merged_framesinfo_labels';


%% read
label_set       = {};
frames_info_set = {};

for i=1:length(input_label_files)
    label_set{i}        = read_label_file(input_label_files{i});
    frames_info_set{i}  = read_txt_file(input_frames_files{i});
end

label       = label_set{1}(1);
frames_info = frames_info_set{1}(1);

%% parse

for i=1:length(label_set)
    
    task_start     = NaN;
    task_end       = NaN;
    task_start_new = NaN;
    task_end_new   = NaN;
    
    for j=1:length(label_set{i})
        
        l = label_set{i}(j);
        
        if strcmp('start', l.name)
            
            task_start     = l.start;
            task_start_new = length(frames_info);
            
            label(end+1) = struct('name', 'start', 'start', task_start_new, 'end', task_start_new);
            
            
        elseif strcmp('end', l.name)
            
            task_end     = l.end;
            task_end_new = task_end - task_start + task_start_new;
            
            label(end+1) = struct('name', 'end', 'start', task_end_new, 'end', task_end_new);
            
            frames_info = [frames_info  frames_info_set{i}(task_start:task_end)];
            
            assert(task_start <= task_end);
            assert(task_start_new <= task_end_new);
            task_start     = NaN;
            task_end       = NaN;
            task_start_new = NaN;
            task_end_new   = NaN;
            
        else
            
            l.start = l.start - task_start + task_start_new;
            l.end   = l.end   - task_start + task_start_new;
            
            label(end+1) = l;
            
        end
        
    end
    
end

label(1) = [];
frames_info(1) = [];

clearvars -except frames_info label output

save(output)










