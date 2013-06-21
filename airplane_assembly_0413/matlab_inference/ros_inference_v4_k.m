
%% load data
addpath(genpath('.'));
addpath('../../cpl_collab_manip/matlab/bin_multistep_plan')
clc; clear; % close all;

init_for_s3 % linear chain
% init_for_s % 3 tasks

m = gen_inference_net(MODEL_PATH);
m.bin_req = bin_req;

% adjust_detection_var; % for adjust detection variance, see that file

%% const

PORT_NUMBER         = 12341;  % must match ROS node param
BIN_NUM             = 20;     % must match ROS node param
MAX_NAME_LENGTH     = 20;     % must match ROS node param
MAX_WS_BINS         = 20;     % must match ROS node param

DO_INFERENCE             = 1;
SEND_INFERENCE_TO_ROS    = 0;

DRAW_DISTRIBUTION_FIGURE = 000;
DRAW_DISTRIBUTION_FIGURE = 399;

DRAW_POSITIONS_FIGURE    = 0;
DRAW_DETECTIONS_FIGURE   = 3310;

DRAW_GT_ACTIONS          = 1;

DRAW_CURRENT_ACTION_PROB = 0; % todo

kelsey_planning = 0;
kelsey_viz      = 1;

%% open connection

ros_tcp_connection                  = tcpip('localhost', PORT_NUMBER);
ros_tcp_connection.OutputBufferSize = 4 * length(m.grammar.symbols) * m.params.T;
ros_tcp_connection.InputBufferSize  = 4 * length(m.grammar.symbols) * m.params.T;
disp('Try connecting to ROS node....');
while 1
    try
        fopen(ros_tcp_connection)
        disp('Connected');
        break;
    catch e
        pause(1);
    end
end

%% init planning

if kelsey_planning
    k = k_planning_init(m);
else
    k = n_planning2_init(m);
end


%% set up variables

t               = 0;
nt              = 0;
inference_num   = 0;

detection_raw_result  = ones(length(m.detection.result), m.params.T);
m.start_conditions(:) = 1;

action_names_gt   = struct([]);
frames_info       = struct([]);
bins_availability = nan(BIN_NUM, m.params.T);

%% LOOP

while t < m.params.T * m.params.downsample_ratio
    
    % exist signal
    if ros_tcp_connection.BytesAvailable == 5
        disp 'Exit signal received'
        break;
    end
    assert(nt < m.params.T - 10);
    
    %------------------------------------------------
    % get new frame data
    %------------------------------------------------
    if ros_tcp_connection.BytesAvailable >= 4 * (2 * 3 + BIN_NUM * 7 + 1 + MAX_WS_BINS)
        
        % new frame, update t
        t   = t + 1;
        nt  =  ceil(t / m.params.downsample_ratio);
        
        % read data from ROS
        rosdata    = fread(ros_tcp_connection, 2 * 3 + BIN_NUM * 7 + 1 + MAX_WS_BINS, 'float');
        
        % read additional data
        len = fread(ros_tcp_connection, 1, 'int');
        additional_data = char(fread(ros_tcp_connection, len, 'char'))';
        additional_data = nx_fromxmlstr(additional_data);
        
        % skip frame?
        if mod(t, m.params.downsample_ratio) ~= 0
            continue
        end
        
        % parse data
        frame_info           = struct;
        frame_info.lefthand  = rosdata(1:3);
        frame_info.righthand = rosdata(4:6);
        for b=0:length(m.detection.detectors)-1
            if m.detection.detectors(b+1).exist
                frame_info.bins(b+1).pq = rosdata(6 + b*7 + 1: 6 + b*7 + 7);
                frame_info.bins(b+1).H  = pq2H(frame_info.bins(b+1).pq);
            end
        end
        if length(frames_info) == 0
            frames_info = frame_info;
        else
            frames_info(end+1) = frame_info;
        end
        ws_bins_data = rosdata(end-MAX_WS_BINS:end);
        ws_bins_len = round(ws_bins_data(1));
        ws_bins = ws_bins_data(2:ws_bins_len+1);
        
        % run detection on new frame
        d = run_action_detections(frame_info, m.detection);
        d(find(isnan(d))) = 1;
        detection_raw_result(:,nt) = d;
        
        % update start condition
        for b=1:length(m.detection.detectors)
            
            % if ~isempty(frame_info.bins(b).H)
            %     % d = norm([-1, -1.3] - [frame_info.bins(b).pq(1), frame_info.bins(b).pq(2)]);
            %     % condition_no = d > 1;
            %     
            % else
            %     condition_no = 1;
            % end
            condition_no = ~any(b == ws_bins); % true if bin not in ws
            
            bins_availability(b,nt) = ~condition_no;
            
            if condition_no
                for i=1:length(m.grammar.symbols)
                    if m.grammar.symbols(i).detector_id == b & m.grammar.symbols(i).is_terminal
                        m.start_conditions(i,nt) = 0;
                    end
                end
            end
           
        end

        % update action labels
        if isfield(additional_data, 'current_action_name')
            
            if length(action_names_gt) == 0
                
                action_names_gt(1).name  = additional_data.current_action_name;
                action_names_gt(1).start = nt;
            
            elseif ~strcmp(action_names_gt(end).name, additional_data.current_action_name)
                
                action_names_gt(end+1).name  = additional_data.current_action_name;
                action_names_gt(end).start   = nt;
            end
        end
        
        continue;
    end
    
    if t <= 0
        continue;
    end
    
    
    
    %------------------------------------------------
    % inference
    %------------------------------------------------
    if DO_INFERENCE
        inference_num = inference_num + 1;
        disp(['Inference num: ' num2str(inference_num)]);
        
        % compute detection result matrix
        for i=1:length(m.detection.result)
            %m.detection.result{i} = triu(repmat(detection_raw_result(i,:)', [1 m.params.T]));
            m.detection.result{i} = repmat(detection_raw_result(i,:)', [1 m.params.T]);
        end
        
        % do inference
        m = m_inference_v3(m);
        
        
        % send to ROS
        if SEND_INFERENCE_TO_ROS
            for i=length(m.grammar.symbols):-1:1
                s = m.grammar.symbols(i);
                name = [s.name '_start'];
                while length(name) < MAX_NAME_LENGTH, name = [name ' ']; end;
                fwrite(ros_tcp_connection, name, 'char');
                fwrite(ros_tcp_connection, s.start_distribution, 'float');

                name = [s.name '_end'];
                while length(name) < MAX_NAME_LENGTH, name = [name ' ']; end;
                fwrite(ros_tcp_connection, name, 'char');
                fwrite(ros_tcp_connection, s.end_distribution, 'float');
            end
        end
        
        
        
        % plot
        if DRAW_DISTRIBUTION_FIGURE > 0
            nx_figure(DRAW_DISTRIBUTION_FIGURE);  subplot(3, 1, 1);
            cla;
            m_plot_distributions(m, DRAW_START_DISTRIBUTION, DRAW_END_DISTRIBUTION);
            hold on; plot([nt nt], [-999 999], 'g'); hold off;
            ylim([0 1.1]);
        end
        
    end
    
    
    %------------------------------------------------
    % planning
    %------------------------------------------------
    if nt > 1 & exist('frame_info')
        if kelsey_planning
            k.action_names_gt = action_names_gt;
            k = k_planning_process(k, m, nt, frame_info, bins_availability, ws_bins, kelsey_viz);
        else
            k = n_planning2_process(k, m, nt, frame_info);
        end
    end
    
    
    % ground truth action label
    if DRAW_DISTRIBUTION_FIGURE & DRAW_GT_ACTIONS
        nx_figure(DRAW_DISTRIBUTION_FIGURE);
        
        if isfield(k, 'executedplan') & isfield(k, 'bestplans') 
            subplot(3, 1, 2);
            xlim([0 m.params.T]);
            plot_plan({k.executedplan k.bestplans{end}});
            %plot_plan({k.executedplan});
        end;
        
        subplot(3, 1, 3);
        cla
        ylim([-1 2]);
        xlim([0 m.params.T]);
        grid on;
        hold on;
        plot([nt nt], [-999 999], 'g');
        for i=1:length(action_names_gt)
            thestart = action_names_gt(i).start;
            theend   = nt;
            if i < length(action_names_gt)
                theend = action_names_gt(i+1).start-1;
            end
            
            thecolor = nxtocolor(actionname2detectorid(action_names_gt(i).name, m.grammar ));
            if isempty(thecolor)
                thecolor = [0 0 0];
            end
            not_perform_action = strcmp(action_names_gt(i).name, 'N/A') | ...
                strcmp(action_names_gt(i).name, 'Complete') | strcmp(action_names_gt(i).name, 'waiting') | ...
                ~isempty(strfind(action_names_gt(i).name, 'Waiting')); 
            
            plot([thestart thestart], [0 nxifelse(not_perform_action, 0, 0.5)], 'color', thecolor);
            plot([thestart theend], [nxifelse(not_perform_action, 0, 0.5) 0], 'color', thecolor);
            if action_names_gt(i).name(end) == '1'
                text(thestart, 1, action_names_gt(i).name);
            end
            text(nt, 0, action_names_gt(end).name);
        end
        hold off;
       
    end
    
    
    %------------------------------------------------
    % misc
    %------------------------------------------------
    if DRAW_DETECTIONS_FIGURE
        nx_figure(DRAW_DETECTIONS_FIGURE);
        dnum = sum([m.detection.detectors.exist]);
        i    = 0;
        for d=1:length(m.detection.detectors)
            if m.detection.detectors(d).exist
                i = i + 1;
                subplot(dnum, 1, i);
                dr = detection_raw_result(d,:);
                plot(dr);
                ylim([0, max(detection_raw_result(:))]);
                legend({['Bin ' num2str(d)]});
            end
        end
        
    end
    
    if exist('frame_info') & DRAW_POSITIONS_FIGURE
        
        nx_figure(DRAW_POSITIONS_FIGURE);
        cla;
        axis equal;
        xlim([-1.5 0.5])
        ylim([-1.5 0.5])
        hold on;
        plot(frame_info.lefthand(1), frame_info.lefthand(2), '*r');
        plot(frame_info.righthand(1), frame_info.righthand(2), '*r');

        for b=0:length(m.detection.detectors)-1
          if ~isempty(frame_info.bins(b+1).H)
            plot(frame_info.bins(b+1).pq(1), frame_info.bins(b+1).pq(2), '.b');
            text(frame_info.bins(b+1).pq(1), frame_info.bins(b+1).pq(2), num2str(b+1)); 
            
            d = max(norm([frame_info.righthand(1), frame_info.righthand(2)] - [frame_info.bins(b+1).pq(1), frame_info.bins(b+1).pq(2)]), norm([frame_info.lefthand(1), frame_info.lefthand(2)] - [frame_info.bins(b+1).pq(1), frame_info.bins(b+1).pq(2)]));
            if d < 0.7
                plot(frame_info.bins(b+1).pq(1), frame_info.bins(b+1).pq(2), '+g');
            end
          end
        end

        hold off
    end
    
    if ~kelsey_planning
        if ~isempty(findall(0,'Type','Figure'))
            pause(1)
        end
    end
    
end


%% close

k = k_planning_terminate(k);

fclose(ros_tcp_connection);

disp 'The End'
disp([num2str(inference_num * 30 / t) ' inferences per second']);


pause(1);




