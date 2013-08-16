
loop_var = 1;
save('loop_var.mat', 'loop_var');

while 1
    load('loop_var.mat')
    if loop_var<=80
        KPH_NOISY=0;
        
    elseif loop_var>80 && loop_var<=160
        KPH_NOISY=1;
    elseif loop_var>160 && loop_var<=240
        KPH_NOISY=-1;    
    else
        exit
    end
    clc; clear; % close all;
    try
    KPH_NOISY = 0;
    kelsey_planning = 1;
    kelsey_viz      = 0;
    NAM_NOISE_MODEL = 0;
    NAM_NOISY = 0;
    ros_inference_v4_k
            %fclose(ros_tcp_connection);
            %disp 'Inference ended'
        
    catch
        %fclose(ros_tcp_connection);
        disp 'Exception!'
    end
    
    %fclose(ros_tcp_connection);
    
    load('loop_var.mat')
    loop_var = loop_var+1;
    save('loop_var.mat', 'loop_var');
    save_batch;
    
    disp 'Start again in 5 seconds...'
    pause(5);
end
    