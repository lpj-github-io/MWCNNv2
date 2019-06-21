clear;close all;

addpath(genpath('./.'));

%% uniform setting

modelname      = 'MWCNN';

scales        = 2;

%% training and testing pair
folder_train  = {'E:\Data_set\DIV2K_train_HR' };     % training


size_input    = 512;          % training
size_label    = 512;          % testing

stride_train  = 500;         % training
stride_test   = 500; %36;          % testing  %%% уе 

val_train     = 0;           % training % default
val_test      = 1;           % testing  % default

training_task = 'denoising';

%% training dataset
[data,  labels,  set]  = patches_generation(scales,size_input,size_label,stride_train,folder_train,val_train,training_task);



%% dataset
images.data   = data;
clear data
images.labels = labels;
clear labels
images.set    = set;
meta.sets = {'train','val','test'};


if ~exist(fullfile('./', modelname),'file')
    mkdir(fullfile('./', modelname));
end

%% save data
save(fullfile(fullfile('./', modelname),['image_' training_task]), 'images', 'meta', '-v7.3')

