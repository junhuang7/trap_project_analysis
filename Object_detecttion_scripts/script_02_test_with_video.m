
version = '3_2';
load_detector = true;
debug = 0;
% in case of debugging or just observing how the model worked:

if debug == 1
    start_frame = 1200;
    end_frame = 3000;
    write_video = 0; % 1 if you want to write a video, 0 if not

else
    start_frame = 1;
    end_frame = [];
    write_video = 1; % 1 if you want to write a video, 0 if not

end

%% Do not modify


video_folder = 'C:\Users\acuna\OneDrive - Universitaet Bern\Coding_playground\Anna_playground\videos\cropped'; % folder where the video is stored
video_name = 'left_cropped_RGB_577_580'; % name of the video
video_format = '.mp4'; % format of the video
videoFile = fullfile(video_folder, [video_name, video_format]); % path to the video

% load the video
vid = VideoReader(videoFile);
if isempty(end_frame)
    end_frame = vid.NumFrames;
end
% load detector
if load_detector == 1
    detector_filename = ['detector_v', (version), '.mat'];
    detector_path = fullfile(root_path, 'detectors');
    load(fullfile(detector_path, detector_filename), 'detector')
end


if write_video == 1
    % Init video Writer
    video_to_write = fullfile(video_folder, [video_name, 'detector_v_', version, video_format]);
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
detectionThreshold = 0.18;
% Network input size
inputSize = detector.TrainingImageSize;
% inputSize = [720 720 3];

figure
% Process video frame by frame
for iframe = start_frame:end_frame

    has_label = 0; % write video for only the ones with labels
    frame = read(vid, iframe);
    % frame = readFrame(vid); % Read the current frame
    resizedFrame = imresize(frame, inputSize(1:2));
    % Perform object detection
    [bboxes, scores, labels] = detect(detector, resizedFrame, MiniBatchSize=8, Threshold=detectionThreshold);
    
    % Filter out detections below the threshold
    validIdx = scores > detectionThreshold;
    % Select only the max score bbox
    if validIdx
        validIdx = find (ismember(scores,max(scores)));
        has_label = 1;
    end

    bboxes = bboxes(validIdx, :);
    labels = labels(validIdx);


    % Store results
    detectedLabels{1,end+1} = labels;
    % store frames in detectedLabels in the second row
    detectedLabels{2,end} = iframe;
    detectedBoxes{end+1} = bboxes;

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
        imshow(annotatedFrame);
        title([num2str(iframe),  '-', labels(1)])
        drawnow; % Update the figure window
        
    end

    
    if write_video == 1 && has_label
        
        % if ~isempty(labels)
        %     title([num2str(iframe),  '-', labels(1)])
        % else
        %     title([num2str(iframe),  '-'])
        % end

       
        % write video 
        frame_to_write = getframe(gca);
        writeVideo(v, frame_to_write);
    end
       
end

if write_video == 1
    % close video
    close(v);
end 

% Save detectors
if ~debug
    detected_labels_filename =  fullfile(root_path, 'stimuli_detected', ['detected_labels_v', version, '.mat']);
    if ~exist(fullfile(root_path, 'stimuli_detected'), 'dir')
        mkdir(fullfile(root_path, 'stimuli_detected'))
    end
    save(detected_labels_filename, "detectedLabels", "detectedBoxes")
    disp(['Labels saved in :', detected_labels_filename])
end
disp('done!')