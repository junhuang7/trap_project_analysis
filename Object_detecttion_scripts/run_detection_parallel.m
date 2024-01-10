clc

% output
% Determine the command window width
commandWindowWidth = matlab.desktop.commandwindow.size;
commandWindowWidth = commandWindowWidth(1);

disp('Initializing parallelization \n')

batchSize = 200 ; % Number of frames to process at once
nBatches = ceil(end_frame / batchSize);
% % Turn on the profiler
% profile on
% Preallocate results (assuming you store results in cells, adjust as needed)
batchResults = cell(nBatches, 1);
% tic
for iBatch = 1:nBatches
    frames = cell(batchSize, 1); % Preallocate cell array for frames
    tempResults = cell(batchSize, 1); % Temporary storage for parfor results
    % Read in the batch of frames
    for iFrame = 1:batchSize
        iframe = (iBatch-1)*batchSize + iFrame;
        if iframe > end_frame
            break;
        end
        frames{iFrame} = readFrame(vid);
    end
    % disp(['Processing Batch: ', num2str(iBatch)])
    % Process each frame in the batch in parallel
    tic
    parfor iFrame = 1:numel(frames)
        if ~isempty(frames{iFrame})
            resizedFrame = imresize(frames{iFrame}, inputSize(1:2));
            % Perform object detection on the resized frame
            [bboxes, scores, labels] = detect(detector, resizedFrame, ...
                                              'MiniBatchSize', 8, ...
                                              'Threshold', detectionThreshold);

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

            labels = labels(validIdx);
            % Store the results
            tempResults{iFrame,1} ={iFrame,labels};
            % tempResults{iFrame,2} =iFrame;
        end
        
    end
     % Assign the temporary results to the batchResults
    batchResults{iBatch,1} = tempResults;
    
    elapsedTime = toc;
    % Clear the previous line by printing backspaces
    fprintf(repmat('\b', 1, commandWindowWidth));

    fprintf('\rProcessing Batch: %d - out of: %d -  Elapsed time is %.6f seconds.', iBatch, nBatches,elapsedTime);
    
end
fprintf('\n');


% Initialize the detectedLabels cell array
detectedLabels = cell(2, 0);

% Loop over each batch to concatenate results
for iBatch = 1:length(batchResults)
    for iResult = 1:length(batchResults{iBatch})
        % Extract the result for the current batch and result index
        currentResult = batchResults{iBatch}{iResult};
        
        % Check if the result is not empty
        if ~isempty(currentResult)
            % Append the labels and frame number to detectedLabels
            detectedLabels{1, end+1} = currentResult{2}; % Labels are assumed to be stored here
            detectedLabels{2, end} = (iBatch - 1) * batchSize + iResult; % Calculate the frame number
        end
    end
end


% toc
% % Turn off the profiler
% profile off
% 
% % View the report
% profile viewer
