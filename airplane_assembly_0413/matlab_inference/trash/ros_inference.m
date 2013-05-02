
%% load data

clc; clear; close all;

load s/learntdata;
%load merged_framesinfo_labels;

%% const

data.compute_joint_dist = 0;

DOWN_SAMPLE_BY_MAX = 1;

MAX_NAME_LENGTH = 20;

PORT_NUMBER     = 12341;
BIN_NUM         = 20;

DO_INFERENCE          = 1;
SEND_INFERENCE_TO_ROS = 1;

DO_PLANNING          = 0;
SEND_PLANNING_TO_ROS = 0;

DO_BIN_AVAILABLE     = 0;

DRAW_DISTRIBUTION       = 0;
DRAW_START_DISTRIBUTION = {'Body', 'Nose_A', 'Wing_AD', 'Tail_AD'};
DRAW_END_DISTRIBUTION   = {};

DRAW_POSITIONS          = 0;
DRAW_DETECTIONS         = 0; % need DOWN_SAMPLE_BY_MAX = 1

DRAW_CURRENT_ACTION_PROB = 0;


%% open connection

all_detections  = ones(length(data.detectors), m.T);

connt                  = tcpip('localhost', PORT_NUMBER);
connt.OutputBufferSize = 8 * 1000;
connt.InputBufferSize  = 8 * 1000;
while 1
    try
        fopen(connt)
        disp('Connected');
        break;
    catch e
        disp('Failed to connect, retry....');
        pause(1);
    end
end

%% planning init
planningconnt                  = tcpip('localhost', 54321);
planningconnt.OutputBufferSize = 8 * 1000;
planningconnt.InputBufferSize  = 8 * 1000;
if DO_PLANNING
while 1
    try
        fopen(planningconnt)
        disp('Connected');
        break;
    catch e
        disp('Failed to connect, retry....');
        pause(1);
    end
end
end

T = data.T;

planning.cache_cost.cost_squareddist    = zeros(2*T + 1, 1);
planning.cache_cost.cost_earlyexpensive = zeros(2*T + 1, 1);
planning.cache_cost.cost_lateexpensive  = zeros(2*T + 1, 1);
for i=1:2*T + 1
    planning.cache_cost.cost_squareddist(i)    = cost_squareddist(i-T-1);
    planning.cache_cost.cost_earlyexpensive(i) = cost_earlyexpensive(i-T-1);
    planning.cache_cost.cost_lateexpensive(i)  = cost_lateexpensive(i-T-1);
    planning.cache_cost.cost_zeros(i)          = cost_zeros(i-T-1);
end

data.planning = planning;

%% create figures
if DRAW_DISTRIBUTION
    figure(999);
end
if DRAW_POSITIONS
    figure(DRAW_POSITIONS);
end
if DRAW_DETECTIONS
    figure(DRAW_DETECTIONS);
end

if DRAW_CURRENT_ACTION_PROB 
    figure(DRAW_CURRENT_ACTION_PROB);
end


%%

if DO_BIN_AVAILABLE 
    m.bin_available = ones(20, m.T) / m.T;
end
bin_available_t = nan(20, 1);

t = 0;

inference_num = 0;

detections = [];

while 1
    
    if t >= m.T * data.downsample_ratio || t > 6000
        break;
    end
    
    
    % get new frame data
    while connt.BytesAvailable >= 4 * (2 * 3 + BIN_NUM * 7)
        
        % read data from ROS
        t          = t + 1;
        nt         =  ceil(t / data.downsample_ratio);
        rosdata    = fread(connt, 2 * 3 + BIN_NUM * 7, 'float');
        
        %disp(['Frame ' num2str(t)]);
        
        if DOWN_SAMPLE_BY_MAX == 0 && mod(t, data.downsample_ratio) ~= 0
            continue
        end
        
        % parse data
        frame_info           = struct;
        frame_info.lefthand  = rosdata(1:3);
        frame_info.righthand = rosdata(4:6);
        for b=0:length(data.detectors)-1
        if data.detectors(b+1).exist
            frame_info.bins(b+1).pq = rosdata(6 + b*7 + 1: 6 + b*7 + 7);
            frame_info.bins(b+1).H  = pq2H(frame_info.bins(b+1).pq);
        end
        end
        
        % evaluator on new frame
        dr = run_action_detections(frame_info, data);
        dr(isnan(dr)) = 1;
        if DRAW_DETECTIONS
            all_detections(:,t) = dr';
        end
        detections(end+1,:) = dr;
        
        % check downsampleratio
        if mod(t, data.downsample_ratio) ~= 0
            continue;
        end
        
        % update obv_duration_likelihood
        detections = max(detections, [], 1);
        for i=1:length(m.g)
            if m.g(i).isterminal
                d = data.grammar.symbols(m.g(i).id).bin_id;
                m.g(i).obv_duration_likelihood(nt,nt:end) = detections(d) * m.g(i).durationmat(nt,nt:end);
            end
        end
        detections  = [];
        
        % update bin_available
        for b=1:length(data.detectors)
          if ~isempty(frame_info.bins(b).H)
            d = norm([-1, -1.3] - [frame_info.bins(b).pq(1), frame_info.bins(b).pq(2)]);
            if d < 1 & isnan(bin_available_t(b))
                bin_available_t(b) = nt;
                disp(['Bin ' num2str(b) ' available']);
            end
          end
        end
        if DO_BIN_AVAILABLE 
            m.bin_available(:) = 0;
            for b=1:length(bin_available_t)
                bat = bin_available_t(b);
                if isnan(bat)
                    bat = nt + 10;
                end
                m.bin_available(b,bat:bat+29) = 1 / 30;
            end
        end
    end
    
    
    
    
    % draw positions
    if t > 10 & DRAW_POSITIONS
        
        set(0,'CurrentFigure',DRAW_POSITIONS)
        cla;
        axis equal;
        xlim([-1.5 0.5])
        ylim([-1.5 0.5])
        hold on;
        plot(frame_info.lefthand(1), frame_info.lefthand(2), '*r');
        plot(frame_info.righthand(1), frame_info.righthand(2), '*r');

        for b=0:length(data.detectors)-1
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

    % draw detections
    if t > 10 & DRAW_DETECTIONS
        set(0,'CurrentFigure',DRAW_DETECTIONS)
        dnum = sum([data.detectors.exist]);
        i    = 0;
        for d=1:length(data.detectors)
            if data.detectors(d).exist
                i = i + 1;
                subplot(dnum, 1, i);
                dr = all_detections(d,:);
                dr(all_detections(d,:) < 0.1) = 0.1;
                semilogy(dr);
                legend({['Bin ' num2str(d)]});
                
                hold on;
                x = (log(min(100, dr(t))) - log(0.1)) / (log(100) - log(0.1));
                semilogy(t, dr(t), '*r');
                set(gca,'Color', x * [1 0.3 0.3] + (1 - x) * [1 1 1]);
                hold off;
            end
        end
        
    end
    
    
    
    % inference
    if t > 0 & DO_INFERENCE
        m = m_inference_v2(m, data, 1);
        data.grammar.symbols = calculate_symbol_distribution(m, data.grammar.symbols);
        inference_num = inference_num + 1;
        disp(['Inference num: ' num2str(inference_num)]);
        
        % print best terminal
        if DRAW_CURRENT_ACTION_PROB
            
            set(0,'CurrentFigure', DRAW_CURRENT_ACTION_PROB)
            a = zeros(1, length(data.grammar.symbols));
            for i=1:length(m.g), 
                if m.g(i).isterminal
                    m.ib.g(i).j = m.ib.g(i).joint2 .* m.if.g(i).joint1;
                    m.ib.g(i).j = m.ib.g(i).j / sum(m.ib.g(i).j(:));
                    m.ib.g(i).j = m.ib.g(i).j * m.ib.g(i).prob_notnull;
                    a(m.ib.g(i).id) = a(m.ib.g(i).id) + sum(sum(m.ib.g(i).j(1:nt,nt:end)));
                end; 
            end;
            barh(a(find([data.grammar.symbols.is_terminal] > 0)));
            labels = {data.grammar.symbols.name};
            labels = labels(find([data.grammar.symbols.is_terminal] > 0));
            set(gca, 'YTick', 1:length(labels), 'YTickLabel', labels);
            
        end
    end
    
    % send to ROS
    if t > 0 & SEND_INFERENCE_TO_ROS
    for i=length(data.grammar.symbols):-1:1

        s = data.grammar.symbols(i);
        name = [s.name '_start'];
        while length(name) < MAX_NAME_LENGTH, name = [name ' ']; end;
        fwrite(connt, name, 'char');
        fwrite(connt, s.start_distribution, 'float');

        name = [s.name '_end'];
        while length(name) < MAX_NAME_LENGTH, name = [name ' ']; end;
        fwrite(connt, name, 'char');
        fwrite(connt, s.end_distribution, 'float');
        
    end
    end
    
    %
    if t > 0 & DO_PLANNING
        do_planning;
    end
    
    
    
    
    % draw inference result
    if t > 0 & DRAW_DISTRIBUTION
        set(0,'CurrentFigure',999)
        cla;
        plot_distribution;
    end
    
    
    if ~isempty(findall(0,'Type','Figure'))
        pause(0.5)
    end
end



fclose(connt);
disp 'The End'
disp([num2str(inference_num * 30 / t) ' inferences per second']);





