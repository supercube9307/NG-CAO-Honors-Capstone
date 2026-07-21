%function Copy_of_LoadInImage()
clc; clear; close all;

ImgsFolder = '/home/super/Documents/NG-CAO-Honors-Capstone/local_data/morph_filter_frames';

% load in the images
FileType = '*.fits';
Img = 'all';
% Get list of all image files
[imgFname,nImgs] = ImgFolderRead(ImgsFolder, FileType, Img);
fpath = fullfile(ImgsFolder, imgFname(1:nImgs));
%[images,~] = imageRead(nImgs,fpath);

%fil = fopen(strcat('..\Full Frame Averaged Brights\', imgFname(20)));
%fil = fread(fil, [1944 2592], 'uint16');
datas = zeros(1944, 2592, nImgs);
%datas = zeros(960, 1296, n Imgs);
for x=1:nImgs
    datas(:, :, x) = fitsread(fpath(x), "Image");
end

source_image = datas(:, :, 1);

%   Random Image Selection
%amount = size(images, 3);
%selection = randi(amount, [1 8]);
%selection = sort(selection);
%image = permute(images(:, :, selection), [2 1 3]);
%image = permute(images(:, :, 3), [2 1 3]);

%   Comment out section if doing block filtering
%{
se = strel('square', 7);
%   Surface of Original Image
figure
surf(image, 'EdgeColor','flat')
%Filtering technique
J = imtophat(image, se);
%   Contrasted Image
figure
imshow(image)
%   Original Image
figure
imshow(image, [])
%   Line Generator
%{
label = 0;
[rows, columns, numberOfColorChannels] = size(image);
for row = 1 : 162 : rows
    yline(row, 'green', label, 'LineWidth', 1);
    label = label + 1;
end
label = 1;
for col = 1 : 216 : columns
    xline(col, 'green', label, 'LineWidth', 1);
    label = label + 1;
end
%}
%   Surface of Filtered Image
figure
surf(J, 'EdgeColor','flat')
%   SNR = (Sum of Signal - Sum of Noise) / Sum of Noise
SNR = sum(J(1097:1101, 178:182), "all") - sum(J(518:522, 73:77), "all");
SNR = SNR / sum(J(518:522, 73:77), "all")
%}

%   Segmentation Start
bim = blockedImage(source_image,BlockSize=[162 216]);
blocks = zeros([162 216 144]);
strel3 = strel('square', 3);
strel4 = strel('square', 4);
strel5 = strel('square', 5);
strel6 = strel('square', 6);
strel7 = strel('square', 7);
strel8 = strel('square', 8);
strel9 = strel('square', 9);
strel10 = strel('square', 10);
strel11 = strel('square', 11);

z = 1;
for x=1:12
    for y = 1:12
        blocks(:, :, z) = getBlock(bim, [x y]);
        z = z + 1;
    end
end

for x = 1:144
    if x==1 || x==2 || x==24 || x==36 || x==38 || x==39 || x==48 || x==50 || x==52 || x==64 || x==69 || x==76 || x==81 || x==86 || x==88 || x==93 || x==108 || x==111 || x==120 || x==132 || x==133 || x==134
        blocks(:, :, x) = imtophat(blocks(:, :, x), strel8);
    elseif x==3 || x==10 || x==11 || x==12 || x==14 || x==15 || x==26 || x==27 || x==35 || x==40 || x==47 || x==53 || x==54 || x==55 || x==56 || x==57 || x==59 || x==65 || x==66 || x==67 || x==68 || x==71 ...
        || x==77 || x==78 || x==80 || x==83 || x==89 || x==90 || x==91 || x==92 || x==95 || x==98 || x==100 || x==101 || x==102 || x==104 || x==105 || x==110 || x==112 || x==135 || x==143 || x==144
        blocks(:, :, x) = imtophat(blocks(:, :, x), strel7);
    elseif x==4 || x==5 || x==7 || x==8 || x==9 || x==16 || x==17 || x==19 || x==20 || x==21 || x==22 || x==23 || x==28 || x==29 || x==31 || x==32 || x==33 || x==34 || x==41 || x==42 || x==44 || x==45 ...
            || x==46 || x==58 || x==70 || x==79 || x==82 || x==94 || x==103 || x==106 || x==107 || x==113 || x==114 || x==116 || x==117 || x==118 || x==119 || x==122 || x==123 || x==136 || x== 140 || x==141 || x==142
        blocks(:, :, x) = imtophat(blocks(:, :, x), strel6);
    elseif x==6 || x==18 || x==30 || x==43 || x==115 || x==124 || x==130 || x==131 || x==137 || x==138 || x==139
        blocks(:, :, x) = imtophat(blocks(:, :, x), strel5);
    elseif x==125 || x==126 || x==128 || x==129
        blocks(:, :, x) = imtophat(blocks(:, :, x), strel4);
    elseif x==127
        blocks(:, :, x) = imtophat(blocks(:, :, x), strel3);
    elseif x==13 || x==25 || x==51 || x==60 || x==62 || x==63 || x==72 || x==74 || x==84 || x==87 || x==96 || x==99 || x==109 || x==121
        blocks(:, :, x) = imtophat(blocks(:, :, x), strel9);
    elseif x==37 || x==75 || x==97
        blocks(:, :, x) = imtophat(blocks(:, :, x), strel10);
    elseif x==49 || x==61 || x==73 || x==85
        blocks(:, :, x) = imtophat(blocks(:, :, x), strel11);
    end
end

destination = [];
bim2 = blockedImage(destination,[1944 2592],[162 216], 0,Mode="w");

z = 1;
for x=1:12
    for y = 1:12
        setBlock(bim2, [x y], blocks(:, :, z))
        z = z + 1;
    end
end

bim2.Mode = "r";
filtered_image = bim2.Source{:};
f1 = figure;
imshow(filtered_image)
title("First Image")

f2 = figure;
imshow(filtered_image, [])
title("Second Image")

scaled_image = (255 * filtered_image) / max(filtered_image(:));
imwrite(cast(scaled_image, 'uint8'), "matlab_output.png");

%   Segmentation End

%{
%   Rotation Check
image = permute(image, [2 1 3]);
image = flip(imagek, 1);
imshow(imagek)
figure
surf(imagek, 'EdgeColor','flat')
K = imtophat(imagek, se);
figure
imshow(K, [])
figure
surf(K, 'EdgeColor','flat')
test = flip(K, 1);
test = permute(test, [2 1 3]);
isequal(J, test)
%}