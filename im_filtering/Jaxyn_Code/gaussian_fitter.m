clear; close all; clc
global frame_to_fit xp yp deltax deltay ifit iter;

% frame = open('J.mat'); 
frame = open('../dot_grid.mat');
square_length = 10;
half = round(square_length/2);
subframe = double(frame.fits(930-half:930+half,810-half:810+half));
% subframe = frame.J(1025-round(square_length/2):1025+round(square_length/2),2185-round(square_length/2):2185+round(square_length/2));
frame_to_fit = subframe; % col 27 of subframe for PSF
initialPars = [300, 15000,0,square_length/2,square_length/2,1,1];
[xp,yp] = meshgrid(1:square_length+1);
options = LMFnlsq;
options = LMFnlsq(options,'FunTol',1e-7,'XTol',1e-7);
ifit = 1; iter = 0;
[par_frameFit, ssq,cnt,nfJ, ~] = LMFnlsq('rotated_gaussian',initialPars,options);

[res, fitted_gaussian] = rotated_gaussian(par_frameFit);
disp(par_frameFit);

figure;
imagesc(subframe);colorbar;
figure;
% look at residual
imagesc(subframe-fitted_gaussian);colorbar;

sigmax = abs(par_frameFit(6));
sigmay = abs(par_frameFit(7));
a=sigmax; % horizontal radius
b=sigmay; % vertical radius
x0=par_frameFit(4); % x0,y0 ellipse centre coordinates
y0=par_frameFit(5);
% rotate the ellipse according to the fitted orientation angle
theta = par_frameFit(3);
X = cos(theta)*(xp-x0) - sin(theta)*(yp-y0);
Y = sin(theta)*(xp-x0) + cos(theta)*(yp-y0);
% get the pixels that fall closest to the ellipses 
%ellipse_pixels1 = (X.^2/a^2 + Y.^2/b^2 > 0.55 & X.^2/a^2 + Y.^2/b^2 < 1.45); 
%ellipse_pixels2 = (X.^2/(2*a)^2 + Y.^2/(2*b)^2 > 0.75 & X.^2/(2*a)^2 + Y.^2/(2*b)^2 < 1.35); 
%ellipse_pixels3 = (X.^2/(3*a)^2 + Y.^2/(3*b)^2 > 0.85 & X.^2/(3*a)^2 + Y.^2/(3*b)^2 < 1.25); 
% actually, just shade in the regions
ellipse_pixels1 = (X.^2/a^2 + Y.^2/b^2 <= 1); 
ellipse_pixels2 = (X.^2/(2*a)^2 + Y.^2/(2*b)^2 <= 1); 
ellipse_pixels3 = (X.^2/(3*a)^2 + Y.^2/(3*b)^2 <=1); 

figure;
imagesc(fitted_gaussian);colorbar;
figure;
subframe1 = subframe;
subframe1(ellipse_pixels1) = inf; % color the ellipses to show up on figure
imagesc(subframe1);colorbar;
figure;
subframe2 = subframe;
subframe2(ellipse_pixels2) = inf;
imagesc(subframe2);colorbar;
figure;
subframe3 = subframe;
subframe3(ellipse_pixels3) = inf;
imagesc(subframe3);colorbar;


% normalized power spectral density of PSF 
PSF_col = subframe(:,half+1);   % column 27 of subframe for this particular PSF
s = size(PSF_col);    
% Length of signal (minus 1 if necessary to make the number even)
L = s(1) -1; 
Fs = 1;     % Sampling frequency: 1 per pixel is all we have                    
f = Fs/L*(0:(L/2));  % frequency domain, just the positive range
% FFT:
Y = abs(fft(PSF_col));
% divide by length of signal which scaled up the signal via fft, and
% multiply by 2 since the amplitude was split across the positives and
% negatives.  Then square it to get the power spectral amplitude
Y = (2*Y/L).^2;
% shift FFT to display positive and negative frequencies in order, and take
% absolute value for real part
Ypos = Y(1:floor(L/2)+1);
figure; semilogy(f, Ypos/sum(abs(Ypos)));

return

%%%%%%%%%%%%%%% Functions Used %%%%%%%%%%%%%%%

function img = clnShow(array,numSigs,optHead,optAxis)
% Input an array, and see it cleanly displayed on a new figure, with no 
% influence on color range by outliers. If you want to add a title, give
% that as the second input. If you want to do things with the axis, such as
% "axis equal" or such, give whatever argument you want to put in axis() as
% the third argument. (If you want to give an axis argument but don't want
% a title, simply give an empty char array, '', as the second argument.)
% ​
if nargin <2
    numSigs = 3;
end
if nargin < 3
    optHead = '';
end
if nargin < 4
    optAxis = "on";
end
statArray = array(:);
preMean = mean(statArray);
stdev = std(statArray);
lowNufs = statArray <= preMean + numSigs*stdev;
lowNuf = statArray(lowNufs);
highNufs = lowNuf >= lowNuf - numSigs*stdev;
clean = lowNuf(highNufs);
avg = mean(clean);
sdev = std(clean);

val = avg - numSigs*sdev;
pek = avg + numSigs*sdev;

figure
img = imagesc(array,[val,pek]);
colorbar; img.Tag = optHead; title(optHead);
axis(optAxis)
end

function y = nPeaks(M_in, n, m)
% find the n highest peaks, assuming that the no other peak exists within 
% m pixels of each found peak 
% B. Nemati, 15-May-2012

[nr,nc] = size(M_in);
M = M_in;
minM = min(min(M));
for ip = 1:n
    maxM = max(max(M));
    [ir, ic] = find(M == maxM, 1, 'first');
    M(max(ir-m,1):min(ir+m,nr),max(ic-m,1):min(ic+m,nc)) = minM;
    peak(ip).value = maxM;
    peak(ip).row = ir;
    peak(ip).col = ic;
end
y = peak;
end