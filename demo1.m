%% Demo to compute P-CNN
% Report bugs to guilhem.cheron@inria.fr
%
% 
% ENABLE GPU support (in my_build.m) and MATLAB Parallel Pool to speed up computation (parpool) 

if ~isdeployed
    addpath('brox_OF'); % Brox 2004 optical flow
end
matconvpath = 'matconvnet-1.0-beta11'; % MatConvNet
run([matconvpath '/my_build.m']); % compile: modify this file to enable GPU support (much faster)
run([matconvpath '/matlab/vl_setupnn.m']) ; % setup  

%% reproduce paper (ICCV 15) results (-0.9% acc, see README.md)
%reproduce_ICCV15_results 

%% P-CNN computation
% ----- PARAMETERS --------
param=[];
%param.lhandposition=13; % pose joints positions in the structure (JHMDB pose format)
param.lhandposition=7;
%param.lhandposition=4;   %CAD60
%param.lhandposition=9;    %MSRDailyActivity3D
%param.rhandposition=12;
param.rhandposition=4;
%param.rhandposition=6;   %CAD60
%param.rhandposition=5;    %MSRDailyActivity3D
%param.upbodypositions=[1 2 3 4 5 6 7 8 9 12 13];
param.upbodypositions=[2 1 3 6 11 6 4 7 5 8];
%param.upbodypositions=[1 2 3 6 5 10 8 7 5 13 12];    %CAD60
%param.upbodypositions=[4 3 2 1 17 13 9 5 8 11 6 17];  %MSRDailyActivity3D
param.lside = 40 ; % length of part box side (also depends on the human scale)
param.savedir = 'p-cnn_features_split1'; % P-CNN results directory
param.impath ='/data/stars/user/sdas/NTU_RGB/images' ; % input images path (one folder per video)
param.imext = '.jpg' ; % input image extension type
param.jointpath = '/data/stars/user/sdas/NTU_RGB/joint_positions' ; % human pose (one folder per video in which there is a file called 'joint_positions.mat')
%param.trainsplitpath = '/data/stars/user/sdas/NTU_RGB/splits/train1.txt'; % split paths
%param.testsplitpath = '/data/stars/user/sdas/NTU_RGB/splits/test1.txt';
param.cachepath = '/data/stars/user/sdas/NTU_RGB/cache'; % cache folder path
param.net_app  = load('models/imagenet-vgg-f.mat') ; % appearance net path
param.net_flow = load('models/flow_net.mat') ; % flow net path
param.batchsize = 128 ; % size of CNN batches
param.use_gpu = false ; % use GPU or CPUs to run CNN?
param.nbthreads_netinput_loading = 20 ; % nb of threads used to load input images
param.compute_kernel = true ; % compute linear kernel and save it. If false, save raw features instead.


% get video names
video_names = dir(param.impath);
video_names={video_names.name};
video_names=video_names(~ismember(video_names,{'.','..'}));

if ~exist(param.cachepath,'dir'); mkdir(param.cachepath) ; end % create cache folder

% 1 - pre-compute OF images for all videos
%parpool('local', 20);
%compute_OF(video_names,param); % compute optical flow between adjacent frames

% 2 - extract part patches
extract_cnn_patches(video_names,param)

% 3 - extract CNN features for each patch and group per video
extract_cnn_features(video_names,param)

% 4 - compute final P-CNN features + kernels
%compute_pcnn_features(param); % compute P-CNN for split 1

%{
% compute for another split
param.savedir = 'p-cnn_features_split2';
param.trainsplitpath =  '/data/stars/user/sdas/CAD120/splits/train2.txt';
param.testsplitpath = '/data/stars/user/sdas/CAD120/splits/test2.txt';
compute_pcnn_features(param); % compute P-CNN for split 2

% compute for another split
param.savedir = 'p-cnn_features_split3';
param.trainsplitpath = '/data/stars/user/sdas/CAD120/splits/train3.txt';
param.testsplitpath = '/data/stars/user/sdas/CAD120/splits/test3.txt';
compute_pcnn_features(param); % compute P-CNN for split 3
% compute for another split
param.savedir = 'p-cnn_features_split4';
param.trainsplitpath = '/data/stars/user/sdas/CAD120/splits/train4.txt';
param.testsplitpath = '/data/stars/user/sdas/CAD120/splits/test4.txt';
compute_pcnn_features(param); % compute P-CNN for split 4


param.savedir = 'p-cnn_features_split5';
param.trainsplitpath = '/data/stars/user/sdas/CAD120/splits/train5.txt';
param.testsplitpath = '/data/stars/user/sdas/CAD120/splits/test5.txt';
compute_pcnn_features(param); % compute P-CNN for split 5

% compute for another split
param.savedir = 'p-cnn_features_split6';
param.trainsplitpath = '/data/stars/user/sdas/MSRDailyActivity3D/splits/train6.txt';
param.testsplitpath = '/data/stars/user/sdas/MSRDailyActivity3D/splits/test6.txt';
compute_pcnn_features(param); % compute P-CNN for split 6
% compute for another split
param.savedir = 'p-cnn_features_split7';
param.trainsplitpath = '/data/stars/user/sdas/MSRDailyActivity3D/splits/train7.txt';
param.testsplitpath = '/data/stars/user/sdas/MSRDailyActivity3D/splits/test7.txt';
compute_pcnn_features(param); % compute P-CNN for split 7

param.savedir = 'p-cnn_features_split8';
param.trainsplitpath = '/data/stars/user/sdas/MSRDailyActivity3D/splits/train8.txt';
param.testsplitpath = '/data/stars/user/sdas/MSRDailyActivity3D/splits/test8.txt';
compute_pcnn_features(param); % compute P-CNN for split 8

% compute for another split
param.savedir = 'p-cnn_features_split9';
param.trainsplitpath = '/data/stars/user/sdas/MSRDailyActivity3D/splits/train9.txt';
param.testsplitpath = '/data/stars/user/sdas/MSRDailyActivity3D/splits/test9.txt';
compute_pcnn_features(param); % compute P-CNN for split 9
% compute for another split
param.savedir = 'p-cnn_features_split10';
param.trainsplitpath = '/data/stars/user/sdas/MSRDailyActivity3D/splits/train10.txt';
param.testsplitpath = '/data/stars/user/sdas/MSRDailyActivity3D/splits/test10.txt';
compute_pcnn_features(param); % compute P-CNN for split 10
%}