function master_map_resized = gbvs(img,param)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                                                     %
% This computes the GBVS map for an image and puts it in master_map.                                  %
%                                                                                                     %
% If this image is part of a video sequence, motionInfo needs to be recycled in a                     %
% loop, and information from the previous frame/image will be used if                                 %
% "flicker" or "motion" channels are employed.                                                        %
% You need to initialize prevMotionInfo to [] for the first frame  (see demo/flicker_motion_demo.m)   %
%                                                                                                     %
%  input                                                                                              %
%    - img can be a filename, or image array (double or uint8, grayscale or rgb)                      %
%    - (optional) param contains parameters for the algorithm (see makeGBVSParams.m)                  %
%                                                                                                     %
%  output structure 'out'. fields:                                                                    %
%    - master_map is the GBVS map for img. (.._resized is the same size as img)                       %
%    - feat_maps contains the final individual feature maps, normalized                               %
%    - map_types contains a string description of each map in feat_map (resp. for each index)         %
%    - intermed_maps contains all the intermediate maps computed along the way (act. & norm.)         %
%      which are used to compute feat_maps, which is then combined into master_map                    %
%    - rawfeatmaps contains all the feature maps computed at the various scales                       %
%                                                                                                     %
%  Jonathan Harel, Last Revised Aug 2008. jonharel@gmail.com                                          %
%                                                                                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

img = double(img)/255;

if (nargin == 1)
    param = makeGBVSParams;
end
[grframe,param] = initGBVS(param,size(img));
prevMotionInfo = [];

%%%%
%%%% STEP 1 : compute raw feature maps from image
%%%%

[rawfeatmaps, ~] = getFeatureMaps( img , param , prevMotionInfo );

%%%%
%%%% STEP 2 : compute activation maps from feature maps
%%%%

mapnames = fieldnames(rawfeatmaps);
mapweights = zeros(1,length(mapnames));
map_types = cell(length(mapnames), 1);
allmaps = cell(length(mapnames), 1);
i = 0;

for fmapi=1:length(mapnames)
    mapsobj = eval( [ 'rawfeatmaps.' mapnames{fmapi} ';'] );
    numtypes = mapsobj.info.numtypes;
    mapweights(fmapi) = mapsobj.info.weight;
    map_types{fmapi} = mapsobj.description;
    for typei = 1 : numtypes
        for lev = param.levels
            i = i + 1;
            [allmaps{i}.map,~] = graphsalapply( mapsobj.maps.val{typei}{lev} , ...
                grframe, param.sigma_frac_act , 1 , 2 , param.tol );
            allmaps{i}.maptype = [ fmapi typei lev ];
        end
    end
end


%%%%
%%%% STEP 3 : normalize activation maps
%%%%

norm_maps = cell(length(mapnames), 1);
for i=1:length(allmaps)
    algtype = 4;
    [norm_maps{i}.map,~] = graphsalapply( allmaps{i}.map , grframe, param.sigma_frac_norm, param.num_norm_iters, algtype , param.tol );
    norm_maps{i}.maptype = allmaps{i}.maptype;
end

%%%%
%%%% STEP 4 : average across maps within each feature channel
%%%%
comb_norm_maps = cell(length(mapnames), 1);
cmaps = cell(length(mapnames), 1);
for i=1:length(mapnames)
    cmaps{i}=0;
end
Nfmap = cmaps;

for j=1:length(norm_maps)
    map = norm_maps{j}.map;
    fmapi = norm_maps{j}.maptype(1);
    Nfmap{fmapi} = Nfmap{fmapi} + 1;
    cmaps{fmapi} = cmaps{fmapi} + map;
end
%%% divide each feature channel by number of maps in that channel

for fmapi = 1 : length(mapnames)
%     if ( param.normalizeTopChannelMaps)
%         algtype = 4;
%         [cmaps{fmapi},~] = graphsalapply( cmaps{fmapi} , grframe, param.sigma_frac_norm, param.num_norm_iters, algtype , param.tol );
%     end
    comb_norm_maps{fmapi} = cmaps{fmapi};
end

%%%%
%%%% STEP 5 : sum across feature channels
%%%%

master_idx = length(mapnames) + 1;
comb_norm_maps{master_idx} = 0;
for fmapi = 1 : length(mapnames)
    comb_norm_maps{master_idx} = comb_norm_maps{master_idx} + cmaps{fmapi} * mapweights(fmapi);
end
master_map = comb_norm_maps{master_idx};
master_map = attenuateBordersGBVS(master_map,4);
master_map = mat2gray(master_map); % NOTE:MAT2GRAY

%%%%
%%%% STEP 6: blur for better results
%%%%
blurfrac = param.blurfrac;
k = mygausskernel( max(size(master_map)) * blurfrac , 2 );
master_map = myconv2(myconv2( master_map , k ),k');
master_map = mat2gray(master_map); % NOTE:MAT2GRAY

% master_map_resized = imresize(master_map,[size(img,1) size(img,2)]);
master_map_resized = mat2gray(imresize(master_map,[size(img,1) size(img,2)])); % NOTE:MAT2GRAY
end