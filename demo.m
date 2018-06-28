% ========================================================================
% Attended Visual Content Degradation based RR-IQA, Version 1.0
% Copyright(c) 2018 Jinjian Wu, Yongxu Liu, Leida Li, and Guangming Shi
% All Rights Reserved.
%
% ----------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is here
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
%----------------------------------------------------------------------
%
% This is an implementation of the algorithm for creating the
% attended visual content degradation based RR-IQA
%
% Please refer to the following papers
%
% Jinjian Wu, Yongxu Liu, Leida Li, and Guangming Shi
% "Attended Visual Content Degradation Based Reduced Reference Image Quality Assessment" 
% IEEE Access, 2018. 
%
%-----------------------------------------------------------------------
%-----------------------------------------------------------------------

close all
clear
clc

% set GBVS environment
gbvsPath = genpath([pwd '/GBVS']);
addpath(gbvsPath);

cd GBVS/
gbvs_install
cd ..

% set HOG bins
numBins = 9;

ref_img = imread('data/I03.BMP');
dst_img = imread('data/i03_11_4.bmp');

% compute 
[fMagr, fSalr] = AVCD( ref_img, numBins );
[fMagd, fSald] = AVCD( dst_img, numBins );

chgMag = (2*fMagr.*fMagd+0.01) ./ (fMagr.^2+fMagd.^2+0.01);
chgSal = (2*fSalr.*fSald+0.01) ./ (fSalr.^2+fSald.^2+0.01);

quality = (mean(chgMag(:))) * (min(chgSal, [], 1));

rmpath(gbvsPath);