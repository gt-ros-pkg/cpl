
while 1
    try
        ros_inference_v4_k
        %fclose(ros_tcp_connection);
        %disp 'Inference ended'
    catch
        %fclose(ros_tcp_connection);
        disp 'Exception!'
    end
    
    %fclose(ros_tcp_connection);
    
    
    save_batch;
    
    disp 'Start again in 5 seconds...'
    pause(5);
end
    