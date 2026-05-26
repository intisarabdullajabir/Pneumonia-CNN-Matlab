function PneumoniaInterfaceImprovedversion

clc;
close all;

%% Load trained model
data = load('ScratchCNN_Improved.mat');
net = data.scratchNet;

%% Create app window
fig = figure( ...
    'Name','Pneumonia Detection System', ...
    'NumberTitle','off', ...
    'Position',[300 150 700 500], ...
    'Color','white');

%% Upload button
uicontrol( ...
    'Style','pushbutton', ...
    'String','Upload Chest X-Ray', ...
    'FontSize',12, ...
    'Position',[250 430 200 40], ...
    'Callback',@uploadImage);

%% Result text
txt = uicontrol( ...
    'Style','text', ...
    'String','Result will appear here', ...
    'FontSize',16, ...
    'FontWeight','bold', ...
    'HorizontalAlignment','center', ...
    'BackgroundColor','white', ...
    'Position',[150 320 400 80]);

%% Image display area
ax = axes( ...
    'Parent',fig, ...
    'Units','pixels', ...
    'Position',[220 30 250 220]);

%% Function executed when button is pressed
    function uploadImage(~,~)

        [file,path] = uigetfile( ...
            {'*.jpg;*.png;*.jpeg'}, ...
            'Select Chest X-Ray');

        if isequal(file,0)
            return;
        end

        %% Read image
        img = imread(fullfile(path,file));

        if size(img,3)==1
            img = cat(3,img,img,img);
        end

        img = imresize(img,[128 128]);

        %% Predict class and scores
        [label,scores] = classify(net,img);

        confidence = max(scores)*100;

        %% Display image
        cla(ax);
        imshow(img,'Parent',ax);

        %% Format label
        lbl = lower(strtrim(string(label)));

        %% Color + output text
        if lbl=="pneumonia"

            txt.String = sprintf( ...
                'PNEUMONIA\nCONFIDENCE: %.2f%%', ...
                confidence);

            txt.ForegroundColor='red';

        elseif lbl=="normal"

            txt.String = sprintf( ...
                'NORMAL\nCONFIDENCE: %.2f%%', ...
                confidence);

            txt.ForegroundColor='green';

        else

            txt.String = sprintf( ...
                'UNKNOWN\nConfidence: %.2f%%', ...
                confidence);

            txt.ForegroundColor='black';

        end

    end

end