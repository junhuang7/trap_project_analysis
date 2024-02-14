% load configs
clear; clc
global GC
GC = general_configs();
close all
version = 'X';
load_detector = true;
debug = 1;
% threshold = 0.145; % 0.18 for v_2
threshold = 0.18; % 0.23

% in case of debugging or just observing how the model worked:
if debug == 1
    start_frame = 5000;
    end_frame = 10000;
    write_video = 0; % 1 if you want to write a video, 0 if not
    fig_visible = true;

    [fi, videos_root]= uigetfile('.mp4', 'Get the video', '');
    d_v ={fi};
    do_parallel = 0;

else
    start_frame = 1;
    end_frame = [];
    write_video = 0; % 1 if you want to write a video, 0 if not
    fig_visible = false;


    % loop through the videos
    videos_root = uigetdir(); % TODO:modify later
    d_v = dir(videos_root);
    d_v = {d_v.name};
    d_v(ismember(d_v, {'.', '..'})) = [];
    d_v = d_v(endsWith(d_v, '.mp4'));
    do_parallel = 1;
   

end
n_vids = length(d_v);

%% Do not modify
% load detector
if load_detector == 1
    %root_path  = 'C:\Users\acuna\OneDrive - Universitaet Bern\Coding_playground\Anna_playground\';
    root_path  = GC.repo_path;

    detector_filename =  ['detector_v', (version), '.mat'];
    detector_path = fullfile(root_path,'Object_detecttion_scripts', 'detectors');
    load(fullfile(detector_path, detector_filename), 'detector')
end

% to fix later
video_folder = videos_root;
for iv = 1:n_vids

    % ask user to input the video
    % [video_name, video_folder] = uigetfile({'*.avi';'*.mp4'}, 'Select the video file');

    %video_folder = 'C:\Users\acuna\OneDrive - Universitaet Bern\Coding_playground\Anna_playground\videos\cropped'; % folder where the video is stored
    %video_name = 'left_cropped_RGB_577_580_less_highlight.mp4'; % name of the video
    video_name = d_v{iv};
    temp_=strsplit(video_name, '.');
    video_filename = temp_{1};

    video_format = '.avi'; % format of the video
    videoFile = fullfile(video_folder, [video_name]); % path to the video

    % load the video
    vid = VideoReader(videoFile);
    
    if isempty(end_frame)
        end_frame = vid.NumFrames;
    end

    

    disp(['Doing - ', video_name])

    if write_video == 1 && iv == 12
        % Init video Writer
        debug_folder = fullfile(video_folder, 'debug');
        if ~exist("debug_folder", "dir")
            mkdir(debug_folder)
        end

        video_to_write = fullfile(debug_folder,[video_name, 'detector_v_', version, video_format]);
        v = VideoWriter(video_to_write, 'MPEG-4');
        open(v);
        disp('## Writing Video ##')
    end

    % Initialize storage for results
    detectedLabels = {};
    detectedBoxes = {};

    % Set the detection threshold
    % detectionThreshold = 0.01;
    % detectionThreshold = 0.179;
    detectionThreshold = threshold;
    % Network input size
    inputSize = detector.InputSize;
    % inputSize = [720 720 3];

    if do_parallel
        run_detection_parallel
    else
        fig = figure('Visible', fig_visible);
        % Process video frame by frame
        tic
        % iframe = 0;
        % while hasFrame(vid)
        for iframe = start_frame:end_frame
            % iframe= iframe+1;

            has_label = 0; % write video for only the ones with labels
            frame = read(vid, iframe);

            % frame = readFrame(vid); % Read the current frame
            resizedFrame = imresize(frame, inputSize(1:2));



            % Perform object detection
            [bboxes, scores, labels] = detect(detector, resizedFrame, MiniBatchSize=8, Threshold=detectionThreshold);

            % check if vF_purple is present. If so, label it as such, because so
            % far, up to v3_2 this is not well recognized
            
            if any(ismember(labels, 'vF_purple')) && any(scores(ismember(labels, 'vF_purple')) > 0.3)
                validIdx = find(ismember(labels, 'vF_purple'));
                if validIdx
                    validIdx = find (ismember(scores,max(scores(ismember(labels, 'vF_purple')))));
                    has_label = 1;
                end
            else


                % Filter out detections below the threshold
                validIdx = scores > detectionThreshold;
                % Select only the max score bbox
                if validIdx
                    validIdx = find (ismember(scores,max(scores)));
                    has_label = 1;
                end
            end
            bboxes = bboxes(validIdx, :);
            labels = labels(validIdx);


            % Store results
            detectedLabels{1,end+1} = labels;
            % store frames in detectedLabels in the second row
            detectedLabels{2,end} = iframe;
            % detectedBoxes{end+1} = bboxes;

            % Optionally, visualize the detection results
            annotatedFrame = insertObjectAnnotation(resizedFrame, 'rectangle', bboxes, cellstr(labels), 'Color', 'yellow');
            % imshow(annotatedFrame);
            % if ~isempty(labels)
            %     title([num2str(iframe),  '-', labels(1)])
            % else
            %     title([num2str(iframe),  '-'])
            % end
            %
            % drawnow; % Update the figure window
            if has_label
                keyboard
                ax=imshow(annotatedFrame);
                title([num2str(iframe),  '-', labels(1)])
                drawnow; % Update the figure window

            end


            if write_video == 1 && has_label && iv ==12

                % if ~isempty(labels)
                %     title([num2str(iframe),  '-', labels(1)])
                % else
                %     title([num2str(iframe),  '-'])
                % end


                % write video
                frame_to_write = getframe(fig);
                writeVideo(v, frame_to_write);
            end

            % if iframe == 57600 % it took 4201 s
            %     keyboard
            % end
        end
        toc
        if write_video == 1 && iv == 12
            % close video
            close(v)
        end
    end
    % Save detectors
    if debug == 0
        % detected_labels_filename =  fullfile('H:\Mario\BioMed_students_2023\Anna\TRAP experiment', 'stimuli_detected', [video_filename,'_','model_v', version, '.mat']);
        eval_pred_root = fullfile(videos_root, 'stimuli_detected');
        if ~exist(eval_pred_root, 'dir')
            mkdir(eval_pred_root)
        end
        detected_labels_filename =  fullfile(eval_pred_root, [video_filename,'_','model_v', version, '.mat']);
        % if ~exist(fullfile(root_path, 'stimuli_detected'), 'dir')
        %     mkdir(fullfile(root_path, 'stimuli_detected'))
        % end
        save(detected_labels_filename, "detectedLabels", "detectedBoxes")
        disp(['Labels saved in :', detected_labels_filename])

       % Run evaluation of predictions, 
       fn_eval_predictions(detectedLabels,video_filename, eval_pred_root)
        
    end
    disp('done!')
end