
loop_var = 1;
save('loop_var.mat', 'loop_var');

while 1
    clc; clear; % close all;
    try
    kelsey_planning = 1;
    kelsey_viz      = 1;
    NAM_NOISE_MODEL = 0;
    NAM_NOISY = 0;
    num_trials = 1;
    
    load('loop_var.mat')
    if loop_var<=num_trials*4
        KPH_NOISY=-1;
    elseif loop_var<=num_trials*4*2
        KPH_NOISY=0;
    else
        exit
    end
    
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
    