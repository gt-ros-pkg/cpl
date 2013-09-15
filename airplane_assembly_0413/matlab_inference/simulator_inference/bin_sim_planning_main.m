
% init_for_s3 % linear chain
%init_for_s % 3 tasks
% init_for_linear_chain_7;
% init_for_linear_chain_robot;
init_for_iros_workshop_2chains_task

m = gen_inference_net(MODEL_PATH);
m.bin_req = bin_req;

% initialze simulator
init_simulator
% humplan is the human duration/bin id action plan
% robplan is the robot's bin remove/delivery action plan

if debug_planning
    fig_planning = figure(101);
    winsize = [560    13   794   935];
    set(fig_planning, 'Position', winsize);

    if create_movie
        % winsize = get(fig_planning, 'Position');
        movie_frames = 200;
        planning_movie = moviein(movie_frames, fig_planning, winsize);
        planning_ind = 1;
    end
end

% adjust_detection_var; % for adjust detection variance, see that file

%% set up variables
nowtimeind = 1;
rate = 30 / m.params.downsample_ratio; 

% TODO FUTURE TIMES ARE == 1
detection_raw_result  = ones(length(m.detection.result), m.params.T);
% m.start_conditions(:) = 1;

T = m.params.T;

% initialize robot/bin environment
[robacts, binstates, binavail, availslot] = gen_rob_bin_states(robplan, bin_init_slots, slots);

while nowtimeind < m.params.T 
    
    if nowtimeind > m.params.T - 30
        'System failure'
        break
    end
    nowtimesec  = nowtimeind / rate;

    % ws_bins is the variable length array of bin IDs which are in the human workspace
    ws_bins = [];
    for bin_ind = 1:numel(binavail)
        if numel(binavail{bin_ind}) > 0 && binavail{bin_ind}(end) == inf
            % bin is available currently 
            ws_bins(end+1) = all_bin_ids(bin_ind);
        end
    end
        
    % get detector data
    % for bin_ind=1:length(m.detection.detectors)
    %     if m.detection.detectors(bin_ind).exist
    %         % I think bin_ind == bin_id
    %     end
    % end
        
    % run detection on new frame
    % d = run_action_detections(frame_info, m.detection);
    % d(find(isnan(d))) = 0;
    % detection_raw_result(:,nowtimeind) = d; % indexed by bin index (frame_info.bins(d).H)
    % TODO: downsample time by m.params.downsample_ratio (rate = 30 / m.params.downsample_ratio)
    % TODO: GEN detection_raw_result

    start_sim = 1

    humacts = gen_reaches(humplan, binavail, availslot, slots);
    if numel(humacts) > 1 && humacts(end).type == 0
        if humacts(end-2).time < nowtimesec
            'Human-robot system completed successfully'
            break
        end
    end
    samp_interval = [0.0,nowtimesec,T/rate];
    samp_num = T;
    detect_dists = sample_detector_dists(humacts, slots, binavail, availslot, samp_interval, ...
                                         samp_num, detector_offset);
    detections = likelihood_function(detect_dists, nowtimeind, likelihood_params);
    
    for bin_ind = 1:numel(all_bin_ids)
        bin_id = all_bin_ids(bin_ind);
        detection_raw_result(bin_id,:) = detections(bin_ind,:);
    end
        
    % update start condition
    m.start_conditions(:) = 0;
    for bin_ind = 1:numel(binavail)
        bin_id = all_bin_ids(bin_ind);
        symbol_inds = [];
        for j = 1:numel(m.grammar.symbols)
            if (m.grammar.symbols(j).is_terminal == 1) && ...
               (m.grammar.symbols(j).detector_id == bin_id) 
               symbol_inds(end+1) = j;
           end
        end
        for t_ind = 2:2:numel(binavail{bin_ind})
            avail_start = binavail{bin_ind}(t_ind-1);
            avail_end = binavail{bin_ind}(t_ind);
            if avail_start == -inf
                avail_start_ind = 1;
            else
                avail_start_ind = ceil((avail_start) * rate)+1;
            end
            if avail_end == inf
                avail_end_ind = nowtimeind;
            else
                avail_end_ind = floor((avail_end) * rate)+1;
            end
            m.start_conditions(symbol_inds,avail_start_ind:avail_end_ind) = 1;
        end
    end
    for j = 1:numel(m.grammar.symbols)
        if ~m.grammar.symbols(j).is_terminal
            m.start_conditions(j,1:nowtimeind) = 1;
        end
    end
    m.start_conditions(:,nowtimeind+1:end) = 1;

    % for bin_id = 1:length(m.detection.detectors)
    %     if ~any(bin_id == ws_bins); % true if bin not in ws
    %         for i=1:length(m.grammar.symbols)
    %             if (m.grammar.symbols(i).is_terminal == 1) && (m.grammar.symbols(i).detector_id == bin_id) 
    %                 m.start_conditions(i,nowtimeind) = 0;
    %             end
    %         end
    %     end
    % end
    
    start_inf = 1
    %------------------------------------------------
    % inference
    %------------------------------------------------
    % compute detection result matrix
    for i=1:length(m.detection.result)
        m.detection.result{i} = repmat(detection_raw_result(i,:)', [1 m.params.T]);
    end
    % do inference
    m = m_inference_v3(m);
    
    %------------------------------------------------
    % planning
    %------------------------------------------------
    start_plan = 1
    lastrminds = -1*ones(1,max(all_bin_ids));
    for bin_ind = 1:numel(binavail)
        if numel(binavail{bin_ind}) > 0 && binavail{bin_ind}(end) < inf
            bin_id = all_bin_ids(bin_ind);
            lastrminds(bin_id) = ceil(rate*binavail{bin_ind}(end));
        end
    end

    bin_distributions = extract_bin_requirement_distributions(m);
    ws_slots = find([slots.row] == 1);
    next_action = bin_simulator_planning(bin_distributions, nowtimeind, ws_bins, ...
                                         lastrminds, robacts, humacts, ws_slots, ...
                                         detection_raw_result, rate, ...
                                         debug_planning, ~viz_extra_info);
    % update robplan sequence with new action
    robplan.times(end+1) = nowtimesec;
    if next_action ~= 0
        robplan.bin_inds(end+1) = find(all_bin_ids == abs(next_action),1);
        if next_action < 0 % remove
            robplan.act_types(end+1) = 2;
        elseif next_action > 0 % deliver
            robplan.act_types(end+1) = 1;
        end
    else
        robplan.bin_inds(end+1) = -1;
        robplan.act_types(end+1) = 0;
    end

    if create_movie
        if planning_ind <= movie_frames
            planning_movie(:,planning_ind) = getframe(fig_planning, winsize);
            planning_ind = planning_ind + 1;
        end
    end

    [robacts, binstates, binavail, availslot] = gen_rob_bin_states(robplan, bin_init_slots, slots);
    if next_action ~= 0
        nowtimeind = ceil(rate * robacts(end-1).end_time);
    else
        nowtimeind = nowtimeind + ceil(rate*3);
    end
end

wait_stats = fluency_measures(humacts)
