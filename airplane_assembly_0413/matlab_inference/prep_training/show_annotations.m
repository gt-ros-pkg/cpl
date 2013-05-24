
clc; clear; 
% close all;

DRAW_3D             = 1;
PAUSE_AT_ANNOTATION = 0;
ONLY_ANNOTATION     = 0;

labels      = read_label_file('../s2/label4.txt');
vid         = VideoReader('../s2/4.avi') 
frames_info = read_txt_file('../s2/4.txt');
startframe  = 1;

%%
for t=1:vid.NumberOfFrames

    % ONLY_ANNOTATION
    if ONLY_ANNOTATION
        annotation_frame = 0;
        for i=1:length(labels)
            if labels(i).start == t,
                annotation_frame = 1;
                break;
            end
        end
        
        if ~annotation_frame,
            continue;
        end
    end
    
    pause(0.01);
    disp(frames_info(t).s);

    % figure
    nx_figure(1); 
    cla;
    hold on;
    if DRAW_3D
        plot3(2, 2, 2, '.');
        plot3(-2,-2,-2, '.');
    end
    axis equal;
    grid on;

    % draw hand
    if DRAW_3D
        plot3(frames_info(t).lefthand(1), frames_info(t).lefthand(2), frames_info(t).lefthand(3), '*r');
        plot3(frames_info(t).righthand(1), frames_info(t).righthand(2), frames_info(t).righthand(3), '*g');
        trplot(eye(4), 'color', 'red');
    else
        plot(frames_info(t).lefthand(1), frames_info(t).lefthand(2), '*r');
        plot(frames_info(t).righthand(1), frames_info(t).righthand(2), '*g');
    end;
    
    for i=1:20
        if ~isempty(frames_info(t).bins(i).H)
            if DRAW_3D
                trplot(frames_info(t).bins(i).H, 'myscale', 0.1, 'frame', num2str(i));
            else
                plot(frames_info(t).bins(i).H(1,4), frames_info(t).bins(i).H(2,4), '*b');
            end
        end
    end
    
    hold off;
    
    % draw video
    nx_figure(2); 
    image(read(vid, t));
    
    % draw action label
    actionlabel = 'None';
    for i=1:length(labels)
        if labels(i).start <= t && labels(i).end >= t
            actionlabel = labels(i).name;
            break;
        end
    end
    text(20, 20, strrep(frames_info(t).s, '_', ''), 'fontSize', 10, 'fontWeight', 'bold', 'BackgroundColor',[.7 .9 .7], 'LineWidth', 1, 'EdgeColor','red');
    text(20, 150, strrep(actionlabel, '_', ''), 'fontSize', 15, 'fontWeight', 'bold', 'BackgroundColor',[.99 .9 .99], 'LineWidth', 1, 'EdgeColor','red', 'Margin',10)

    % pause at annotation
    if PAUSE_AT_ANNOTATION
        for i=1:length(labels)
            if labels(i).start == t,
                pause;
            end
        end
    end
end














