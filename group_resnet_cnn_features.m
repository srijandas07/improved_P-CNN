param.impath ='/data/stars/user/sdas/CAD60new/images' ; % input images path (one folder per video)
param.imext = '.png' ; % input image extension type
param.cachepath = '/data/stars/user/sdas/CAD60new/cache_resnet'; % cache folder path

video_names = dir(param.impath);
video_names={video_names.name};
video_names=video_names(~ismember(video_names,{'.','..'}));

extract_resnet_cnn_features(video_names,param)