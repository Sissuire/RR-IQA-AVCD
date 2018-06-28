function [fMag, fSal] = AVCD( img0, numBins )

% calculate saliency via GBVS (features: RID)
sal = gbvs(imresize(img0, 1/2));
sal = mat2gray( sal );

% downsample
if ndims( img0 ) == 3
    imgg = double( rgb2gray( img0 ) );
else
    imgg = double( img0 );
end
img_filt = imfilter(imgg, fspecial('average', 2), 'same');
img = img_filt(1:2:end, 1:2:end);

% extract OSVP based on saliency weighting.
r = 1;
fMag = func_osvp_sal( img, r, sal );

% extract Saliency Global Feature with HOG.
sal = imresize(sal, size(img));
fSal = func_extractHOGFeatures( sal, numBins );
end

function osvp_hist_m = func_osvp_sal( img, r, sal )

% traditional OVSP calculation.
nnum = 2*r+1;
otr = 6; % threshold for the similar preferred orientation
kx = [ -1 0 1; -1 0 1; -1 0 1 ]/3;
ky = kx';

bin_num = nnum*nnum;
[xg, yg] = meshgrid(0:nnum-1, 0:nnum-1);
rot = [xg(:), yg(:)];
rot( rot(:,1)==r & rot(:,2)==r, : ) = [];

imgd = padarray( img, [r r], 'symmetric' );
[ row, col ] = size( imgd );
Gx = imfilter( imgd, kx );
Gy = imfilter( imgd, ky );
Cimg = sqrt( Gx.^2 + Gy.^2 );
Oimg = atan2(Gy,Gx)/pi*180; % convert to degree measure.

Cvimg = zeros( row,col );
threMag = 10; % mag threshold
Cvimg( Cimg>=threMag ) = 1;
Cvimgc = Cvimg( r+1:row-r,r+1:col-r ); % cut the replicated edge
Oimgc = Oimg( r+1:row-r,r+1:col-r );

ssr_val = zeros( row-2*r, col-2*r );
for i = 1:bin_num-1
    dx = rot(i, 1); dy = rot(i, 2);
    Cvimgn = Cvimg( dx+1:row-2*r+dx, dy+1:col-2*r+dy );
    Oimgn = Oimg( dx+1:row-2*r+dx, dy+1:col-2*r+dy );
    Odif = abs( Oimgn - Oimgc );  % the diff between each neighbour.
    Odif( Odif>180 ) = 360 - Odif( Odif>180 );
    Odif( Cvimgc==0 ) = 360; % the original image mag is lower than threshold.
    last_state = zeros( row-2*r, col-2*r );
    last_state( Cvimgn+Cvimgc==0 ) = 1;
    last_state( Odif<=otr ) = 1;
    ssr_val = ssr_val + last_state;
end

% calculate the gaussian magnitude normalized with LoG.
mag_gmn = func_GMLOG( img );
mag_map = padarray(mag_gmn, [2 2]);

% % histogram of saliency on each visual pattern.
% osvp_hist_s = zeros( bin_num, 1 );
% for i = 1 : bin_num
% %     osvp_hist_s(i) = sum( sum( sal( ssr_val==i-1) ) );
%     osvp_hist_s(i) = sum( sum( sal( ssr_val==i-1) ) ) / (sum(sum(ssr_val == i-1 )) + 0.00001);
% end

% quantization
sal = imresize(sal, size(img));
step = 0.1;
sal = ceil(sal/step)*step;


% histogram of weighted mag on each visual pattern.
mag_map = mag_map.*sal;
osvp_hist_m = zeros( bin_num, 1 );
for i = 1 : bin_num
    osvp_hist_m(i) = sum( sum( mag_map( ssr_val==i-1 ) ) );
end

% combination. (only 9 values as reference features.)
% osvp_hist = osvp_hist_m.^1.7 .* osvp_hist_s.^0.56;

end

function grad_im = func_GMLOG( img )

sigma = 0.5;
[gx,gy] = gaussian_derivative(img,sigma);
grad_im = sqrt(gx.^2+gy.^2);

window2 = fspecial('log', 2*ceil(3*sigma)+1, sigma);
window2 =  window2/sum(abs(window2(:)));
log_im = abs(filter2(window2, img, 'same'));        %% Laplacian of Gaussian.

ratio = 5.4; % default value 2.5 is the average ratio of GM to LOG on LIVE database,
% but higher value may cause a better performance,
% due to its characteristic of weighting.
grad_im = abs(grad_im/ratio);

%Normalization
c0 = 4*0.05;
sigmaN = 2*sigma;
window1 = fspecial('gaussian',2*ceil(3*sigmaN)+1, sigmaN);
window1 = window1/sum(window1(:));

Nmap = sqrt(filter2(window1,mean(cat(3,grad_im,log_im).^2, 3),'same'))+c0;
grad_im = (grad_im)./Nmap;

% remove the borders, which may be the wrong results of a convolution operation
h = ceil(3*sigmaN);
grad_im = abs(grad_im(h:end-h+1,h:end-h+1,:));

end

function [gx,gy] = gaussian_derivative(img,sigma)
window1 = fspecial('gaussian',2*ceil(3*sigma)+1+2, sigma);
winx = window1(2:end-1,2:end-1)-window1(2:end-1,3:end);winx = winx/sum(abs(winx(:)));
winy = window1(2:end-1,2:end-1)-window1(3:end,2:end-1);winy = winy/sum(abs(winy(:)));
gx = filter2(winx,img,'same');
gy = filter2(winy,img,'same');
end

function gradientHistogram = func_extractHOGFeatures( sal, numBins )
% numBins = 9;
dx = [-1, -1, -1; 0, 0, 0; 1, 1, 1];
dy = dx';
Gx = imfilter(sal, dx); Gy = imfilter(sal, dy);
sMag = hypot(Gx, Gy);
sDir = atan2d(-Gy, Gx);
sBin = func_dir2bin(sDir, numBins);
gradientHistogram = zeros(numBins, 1);
for i = 1:numBins
    gradientHistogram(i, 1) = sum(sum( sMag( sBin==i ) ));
end

% salFeatures = [gradientHistogram, histCnt];
end

function bin = func_dir2bin(dir, numBins)

histRange = double(360);
negDir = dir < 0;
dir(negDir) = histRange + dir(negDir);
dir(dir>359.999) = 0.0;

binWidth = histRange/numBins;
invWidth = 1./binWidth;
bin = floor(dir.*invWidth) + 1;
end