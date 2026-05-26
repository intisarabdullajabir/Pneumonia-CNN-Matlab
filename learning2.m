%% =========================================
% PNEUMONIA DETECTION CNN (IMPROVED VERSION)
% ==========================================

clc;
clear;
close all;

%% =========================
% LOAD DATASET
% ==========================

datasetPath = 'C:\Users\jabir\OneDrive\Desktop\chest_xray';

imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

disp('Dataset Summary:')
disp(countEachLabel(imds));

%% =========================
% STRATIFIED DATA SPLITTING
% 70% TRAIN
% 15% VALIDATION
% 15% TEST
% ==========================

[imdsTrain, imdsTemp] = splitEachLabel(imds, ...
    0.7, 'randomized');

[imdsValidation, imdsTest] = splitEachLabel(imdsTemp, ...
    0.5, 'randomized');

disp('Training Data:')
disp(countEachLabel(imdsTrain));

disp('Validation Data:')
disp(countEachLabel(imdsValidation));

disp('Testing Data:')
disp(countEachLabel(imdsTest));

%% =========================
% IMAGE SIZE
% ==========================

imageSize = [128 128 3];

%% =========================
% DATA AUGMENTATION
% (TRAINING ONLY)
% ==========================

augmenter = imageDataAugmenter( ...
    'RandRotation',[-10 10], ...
    'RandXTranslation',[-5 5], ...
    'RandYTranslation',[-5 5]);

%% TRAINING DATA
augTrain = augmentedImageDatastore( ...
    imageSize(1:2), ...
    imdsTrain, ...
    'DataAugmentation', augmenter, ...
    'ColorPreprocessing', 'gray2rgb');

%% VALIDATION DATA
augValidation = augmentedImageDatastore( ...
    imageSize(1:2), ...
    imdsValidation, ...
    'ColorPreprocessing', 'gray2rgb');

%% TEST DATA
augTest = augmentedImageDatastore( ...
    imageSize(1:2), ...
    imdsTest, ...
    'ColorPreprocessing', 'gray2rgb');

%% =========================
% CNN ARCHITECTURE
% ==========================

layers = [

    imageInputLayer([128 128 3], ...
    'Normalization','zerocenter')

    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer

    dropoutLayer(0.5)

    fullyConnectedLayer(2)

    softmaxLayer
    classificationLayer
];



%% =========================
% TRAINING OPTIONS
% ==========================


options = trainingOptions( ...
    'adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augValidation, ...
    'ValidationFrequency', 10, ...
    'Verbose', true, ...
    'Plots', 'training-progress' ...
    );
%% =========================
% TRAIN CNN
% ==========================

disp('Training CNN Model...')

scratchNet = trainNetwork( ...
    augTrain, ...
    layers, ...
    options);

%% =========================
% SAVE MODEL
% ==========================

save('ScratchCNN_Improved.mat', 'scratchNet');

disp('Model Saved Successfully')

%% =========================clc;
clear;
close all;

%% Load Model
data = load('ScratchCNN_Improved.mat');
net = data.scratchNet;

%% Upload Image
[file,path] = uigetfile({'*.jpg;*.png;*.jpeg'}, ...
    'Select Chest X-ray Image');

if isequal(file,0)
    disp('No Image Selected');
    return;
end

%% Read Image
img = imread(fullfile(path,file));
img = imresize(img,[128 128]);

if size(img,3)==1
    img = cat(3,img,img,img);
end

%% Predict
[label,scores] = classify(net,img);

%% REAL Confidence
confidence = scores(label) * 100;

%% Display Image
figure;
imshow(img);

%% Clean label
lbl = lower(strtrim(string(label)));

%% Color-coded output
if lbl == "pneumonia"
    title(sprintf('PNEUMONIA (%.2f%%)', confidence), 'Color','red');

elseif lbl == "normal"
    title(sprintf('NORMAL (%.2f%%)', confidence), 'Color','green');

else
    title(sprintf('%s (%.2f%%)', string(label), confidence), 'Color','black');
end

%% Console output
fprintf('Prediction: %s\n', char(label));
fprintf('Confidence: %.2f%%\n', confidence);

% TEST MODEL
% ==========================

disp('Evaluating Model...')

predictedLabels = classify(scratchNet, augTest);

actualLabels = imdsTest.Labels;

%% =========================
% ACCURACY
% ==========================

accuracy = mean(predictedLabels == actualLabels) * 100;

fprintf('\nTest Accuracy = %.2f%%\n', accuracy);

%% =========================
% CONFUSION MATRIX
% ==========================

figure;
confusionchart(actualLabels, predictedLabels);

title('Confusion Matrix');

%% =========================
% PRECISION / RECALL / F1
% ==========================

confMat = confusionmat(actualLabels, predictedLabels);

TP = confMat(2,2);
TN = confMat(1,1);
FP = confMat(1,2);
FN = confMat(2,1);

precision = TP / (TP + FP);

recall = TP / (TP + FN);

specificity = TN / (TN + FP);

f1score = 2 * ((precision * recall) / ...
    (precision + recall));

fprintf('\nPrecision = %.2f%%\n', precision*100);

fprintf('Recall (Sensitivity) = %.2f%%\n', ...
    recall*100);

fprintf('Specificity = %.2f%%\n', ...
    specificity*100);

fprintf('F1 Score = %.2f%%\n', ...
    f1score*100);

%% =========================
% ROC CURVE
% ==========================

[labelScores] = predict(scratchNet, augTest);

positiveClassScores = labelScores(:,2);

[X,Y,~,AUC] = perfcurve( ...
    actualLabels, ...
    positiveClassScores, ...
    'PNEUMONIA');

figure;
plot(X,Y,'LineWidth',2);

xlabel('False Positive Rate');
ylabel('True Positive Rate');

title(['ROC Curve (AUC = ' num2str(AUC) ')']);

grid on;

%% =========================
% SAMPLE PREDICTIONS
% ==========================

figure;

perm = randperm(numel(imdsTest.Files),9);

for i = 1:9

    subplot(3,3,i)

    img = readimage(imdsTest, perm(i));

    if size(img,3)==1
        img = cat(3,img,img,img);
    end

    imgResized = imresize(img,[128 128]);

    [label,scores] = classify( ...
        scratchNet, ...
        imgResized);

    confidence = max(scores)*100;

    imshow(img)

    title( ...
        sprintf('%s\n%.2f%%', ...
        string(label), confidence));

end

sgtitle('Sample Test Predictions');

disp('Project Completed Successfully')