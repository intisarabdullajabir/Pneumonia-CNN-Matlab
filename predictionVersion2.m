clc;
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
