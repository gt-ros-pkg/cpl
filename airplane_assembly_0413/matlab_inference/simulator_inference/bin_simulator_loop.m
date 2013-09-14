
debug_planning = 1;
create_movie = 0;

if create_movie
    fig_planning = figure(101);
    winsize = get(fig_planning, 'Position');
    movie_frames = 200;
    planning_movie = moviein(movie_frames, fig_planning, winsize);
    planning_ind = 1;
end

%% load data
addpath(genpath('.'));
addpath('../../cpl_collab_manip/matlab/bin_multistep_plan')
addpath('../../cpl_collab_manip/matlab/bin_simulator')

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

% adjust_detection_var; % for adjust detection variance, see that file

%% set up variables
nowtimeind = 1;
rate = 30 / m.params.downsample_ratio; 

% TODO FUTURE TIMES ARE == 1
detection_raw_result  = ones(length(m.detection.result), m.params.T);
m.start_conditions(:) = 1;

T = m.params.T;

% initialize robot/bin environment
[robacts, binstates, binavail, availslot] = gen_rob_bin_states(robplan, bin_init_slots, slots);

while nowtimeind < m.params.T 
    
    assert(nowtimeind < m.params.T - 10);
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
    for bin_id = 1:length(m.detection.detectors)
        if ~any(bin_id == ws_bins); % true if bin not in ws
            for i=1:length(m.grammar.symbols)
                if (m.grammar.symbols(i).is_terminal == 1) && (m.grammar.symbols(i).detector_id == bin_id) 
                    m.start_conditions(i,nowtimeind) = 0;
                end
            end
        end
    end
    
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
    next_action = bin_simulator_planning(bin_distributions, nowtimeind, ws_bins, ...
                                         lastrminds, detection_raw_result, rate, debug_planning);
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

