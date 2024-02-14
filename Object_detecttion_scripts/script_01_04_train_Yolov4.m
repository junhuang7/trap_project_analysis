basenet = resnet50;
%analyzeNetwork(basenet)

% make new layer and set normalization to none
imageSize = basenet.Layers(1).InputSize;
layerName = basenet.Layers(1).Name;
newinputLayer = imageInputLayer(imageSize,'Normalization','none','Name',layerName);

% Extract the layer graph of the base network to use for creating YOLO v4 deep learning network..
lgraph = layerGraph(basenet);
lgraph = removeLayers(lgraph,'ClassificationLayer_fc1000');
lgraph = replaceLayer(lgraph,layerName,newinputLayer);

% create a dlnetwork
dlnet = dlnetwork(lgraph);

% dspecify the name sof the feature extraction layers in the base network,
% to use as heads
featureExtractionLayers = ["activation_1_relu",  "activation_10_relu" ,...
    "activation_22_relu","activation_40_relu",...   
   ];

% specify classes
% Set parameters for the network
class_names = [categorical({'vF_purple'}),...
                categorical({'cold'}), ...
                categorical({'hot'}), ...
                categorical({'vF_blue'}), ...
                categorical({'vF_green'}), ...
                categorical({'pinprick'})];

% Preprocess the data
% Transform the DS
preprocessedData = transform(dsCombined, @(data)resizeImageAndLabel(data, imageSize));

% get anchor boxes 
numAnchors = 4;
aboxes = estimateAnchorBoxes(preprocessedData, numAnchors);
aboxes = arrayfun(@(i) aboxes(i,:), 1:size(aboxes, 1), 'UniformOutput', false)';

% create a detector based on the specified network
detector = yolov4ObjectDetector(dlnet,class_names,aboxes,DetectionNetworkSource=featureExtractionLayers);

[detector,info] = trainYOLOXObjectDetector(augmentedTrainingData,detector,options);