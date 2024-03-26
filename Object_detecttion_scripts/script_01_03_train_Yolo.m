% this script will evaluate the YoloX network in data from videos already
% concatenated. 
clc
clear
close all
global GC
% GC = general_configs();
%% set the version
version = 'X';
rootpath = GC.repo_path; 
% if ispc
%     root_path = 'C:\Users\acuna\OneDrive - Universitaet Bern\Coding_playground\Anna_playground\';
% else
%     keyboard
% end

addpath(genpath('Object_detection_scripts\utilities'))

% load the ground truths
gt_path = 'D:\stimulus_labeler';
% gt_path = 'H:\Mario\BioMed_students_2023\Anna\Playground\YOLO_paincohort';
% check how many *.mat files are in the folder
files = dir(fullfile(gt_path, '*.mat'));
n_files = length(files);

% Load the GT
% load(fullfile(gt_path, ['gTruth_', num2str(i), '.mat']));
load(fullfile(gt_path, ['gTruth_conc.mat']));
% Set where to save the images
write_loc = fullfile(gt_path, ['img', num2str(i)]);
if ~exist(write_loc, 'dir')
    mkdir(write_loc)
end

% Make training data
[imds, bxds] = objectDetectorTrainingData(gTruth, 'WriteLocation', ...
    write_loc); % Unique WriteLocation for each

% Combine the final image and box label datastores for training
dsCombined = combine(imds, bxds);

% Set parameters for the network
class_names = [categorical({'vF_purple'}),...
                categorical({'cold'}), ...
                categorical({'hot'}), ...
                categorical({'vF_blue'}), ...
                categorical({'vF_green'}), ...
                categorical({'pinprick'})];

% Init detector
pretrainedDetector = yoloxObjectDetector("small-coco", class_names);
inputSize = pretrainedDetector.InputSize;

% Transform the DS
preprocessedData = transform(dsCombined, @(data)resizeImageAndLabel(data, inputSize));


% % preview data
% data = preview(preprocessedData);
% I = data{1};
% bbox = data{2};
% label = data{3};
% imshow(I)
% showShape("rectangle", bbox, Label=label)

%% Modify data
% shuffle data for training
rng(0);
preprocessedData = shuffle(preprocessedData);
keyboard
totalSamples = numel(preprocessedData.UnderlyingDatastores{1, 1}.UnderlyingDatastores{1, 1}.Files); % Replace with appropriate method if needed

% Define your split percentages (e.g., 70% training, 15% validation, 15% testing)
trainingSplit = 0.80;
validationSplit = 0.20;
% Test split will be the remaining percentage

% Calculate the number of samples for each subset; check if test is needed
numTrainingSamples = round(totalSamples * trainingSplit);
numValidationSamples = round(totalSamples * validationSplit);
% numTestingSamples = totalSamples - numTrainingSamples - numValidationSamples;

% Generate random indices for each subset
indices = randperm(totalSamples);
training_idx = indices(1:numTrainingSamples);
val_idx = indices(numTrainingSamples+1 : numTrainingSamples+numValidationSamples);
% test_idx = indices(numTrainingSamples+numValidationSamples+1 : end);

% Now you can create subsets
dsTrain = subset(preprocessedData, training_idx);
dsVal = subset(preprocessedData, val_idx);
% dsTest = subset(preprocessedData, test_idx);

% Augment the data
augmentedTrainingData = transform(dsTrain, @augmentData);

% options for YoloX
options = trainingOptions("sgdm", ... % adam is not better
    InitialLearnRate=5e-4, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropFactor=0.99, ...
    LearnRateDropPeriod=1, ...   
    MiniBatchSize=20, ...
    MaxEpochs=100, ...
    BatchNormalizationStatistics="moving", ...
    ExecutionEnvironment="gpu", ...
    Shuffle="every-epoch", ...
    VerboseFrequency=25, ...
    ValidationFrequency=100, ...
    ValidationData=dsVal, ...
    ResetInputNormalization=false, ...
    OutputNetwork="best-validation-loss", ...
    GradientThreshold=30, ...
    L2Regularization=5e-4);


% Train the YOLO X detector.
[detector,info] = trainYOLOXObjectDetector(augmentedTrainingData,pretrainedDetector,options);

%% SAVE
% save detector
detector_filename = ['detector_v', (version), '.mat'];
detector_path = fullfile(rootpath, 'detectors');
if ~exist(detector_path, 'dir')
    mkdir(detector_path)
end
save(fullfile(detector_path, detector_filename), 'detector', 'info')



%% helper functions
function B = augmentData(A)
    addpath('C:\Users\acuna\Documents\MATLAB\Examples\R2023b\deeplearning_shared\MulticlassObjectDetectionUsingDeepLearningExample')
    
    % Apply random horizontal flipping, and random X/Y scaling. Boxes that get
    % scaled outside the bounds are clipped if the overlap is above 0.25. Also,
    % jitter image color.
    % A = readall(A);
    B = cell(size(A));
    % loop throught the rows
    a = 0; b=0.2; % set min and max for variables
    for ir = 1:size(A,1)
        I = A{ir,1};
        sz = size(I);
        r = a + (b-a).*rand(4,1);
        if numel(sz)==3 && sz(3) == 3
            I = jitterColorHSV(I,...
                Contrast=r(1),...
                Hue=r(2),...
                Saturation=r(3),...
                Brightness=r(4));
        end
    
        % % Add randomized Gaussian blur
        I = imgaussfilt(I,1.5*rand);
        %
        % % Add salt and pepper noise
        % I = imnoise(I,"salt & pepper");
    
        % Randomly flip and scale image.
        tform = randomAffine2d(XReflection=true, Scale=[0.95 1.1]);
        rout = affineOutputView(sz, tform, BoundsStyle="CenterOutput");
        B{ir, 1} = imwarp(I, tform, OutputView=rout);
    
        % Sanitize boxes, if needed. This helper function is attached as a
        % supporting file. Open the example in MATLAB to open this function.
        A{ir,2} = helperSanitizeBoxes(A{ir,2});
    
        % Apply same transform to boxes.
        [B{ir,2},indices] = bboxwarp(A{ir,2}, tform, rout, OverlapThreshold=0.25);
        B{ir,3} = A{ir,3}(indices);
    
        % Return original data only when all boxes are removed by warping.
        if isempty(indices)
            B(ir,:) = A(ir,:);
        end
    end
end

function data = resizeImageAndLabel(data,targetSize)
    % Resize the images and scale the corresponding bounding boxes.
    
    for ir = 1:size(data,1)


        scale = (targetSize(1:2))./size(data{ir, 1},[1 2]);
        data{ir,1} = imresize(data{ir, 1},targetSize(1:2));
        data{ir,2} = bboxresize(data{ir,2},scale);

        data{ir,2} = floor(data{ir, 2});
        imageSize = targetSize(1:2);
        boxes = data{ir, 2};
        % Set boxes with negative values to have value 1.
        boxes(boxes<=0) = 1;

        % Validate if bounding box in within image boundary.
        boxes(:,3) = min(boxes(:,3),imageSize(2) - boxes(:,1)-1);
        boxes(:,4) = min(boxes(:,4),imageSize(1) - boxes(:,2)-1);

        data{ir,2} = boxes; 
    end
end

%{
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OLD CODE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% get the layers
featureLayer = "leaky_relu_5";

numAnchors = 9;
aboxes = estimateAnchorBoxes(preprocessedData, numAnchors);

% get the classes
ls = bxds.LabelData(:,2);
catArray = vertcat(ls{:});
class_labels = unique(catArray);


numClasses = length(class_labels);

% Network
pretrainedNet = pretrainedDetector.Network;
lgraph = yolov2Layers(inputSize, numClasses, aboxes, pretrainedNet, featureLayer);

% shuffle data for training
rng(0);
preprocessedData = shuffle(preprocessedData);

totalSamples = numel(imds.Files); % Replace with appropriate method if needed

% Define your split percentages (e.g., 70% training, 15% validation, 15% testing)
trainingSplit = 0.80;
validationSplit = 0.20;
% Test split will be the remaining percentage

% Calculate the number of samples for each subset; check if test is needed
numTrainingSamples = round(totalSamples * trainingSplit);
numValidationSamples = round(totalSamples * validationSplit);
% numTestingSamples = totalSamples - numTrainingSamples - numValidationSamples;

% Generate random indices for each subset
indices = randperm(totalSamples);
training_idx = indices(1:numTrainingSamples);
val_idx = indices(numTrainingSamples+1 : numTrainingSamples+numValidationSamples);
% test_idx = indices(numTrainingSamples+numValidationSamples+1 : end);

% Now you can create subsets
dsTrain = subset(preprocessedData, training_idx);
dsVal = subset(preprocessedData, val_idx);
% dsTest = subset(preprocessedData, test_idx);

% Augment the data
augmentedTrainingData = transform(dsTrain, @augmentData);
data = read(augmentedTrainingData);
I = data{1};
bbox = data{2};
label = data{3};
imshow(I)
showShape("rectangle", bbox, Label=label)

% Set parameters for Training
opts = trainingOptions("rmsprop",...
        InitialLearnRate=0.001,...
        MiniBatchSize=4,...
        MaxEpochs=100,...
        LearnRateSchedule="none",... %'piecewise'
        LearnRateDropPeriod=5,...
        VerboseFrequency=30, ...
        L2Regularization=0.001,...
        ValidationData=dsVal, ...
        ValidationFrequency=50, ...
        ExecutionEnvironment= 'gpu',...
        Plots='training-progress',...
        OutputNetwork="best-validation-loss");


% Train detector
[detector, info] = trainYOLOv2ObjectDetector(augmentedTrainingData,lgraph, opts);

% save detector
detector_filename = ['detector_v', (version), '.mat'];
detector_path = fullfile(root_path_data, 'detectors');
if ~exist(detector_path, 'dir')
    mkdir(detector_path)
end
save(fullfile(detector_path, detector_filename), 'detector', 'info') 
%}

