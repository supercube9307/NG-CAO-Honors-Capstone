function [images,TimeStamp] = imageRead(nImgs,fpath,RecordTime)
%**************************************************************************
% Code to read in image files from spectratrac units
% 3/29/2024 - Bradford Snios, Colin Huber
%
% Input
%   nImgs (1 x 1) double, number of images to load in
%   fpath (1 x nImgs) string, list of filenames to read in
%   RecordTime [1,0] double, whether to record the time stamps
%       1 - Record timestamps
%       0 - Do not record timestamps
% Output
%   images (1336 x 980 x nImgs)
%       double, image data read in from the files

% If RecordTime is not requested, set bool to zero.
if exist('RecordTime','var') == 0 || isempty(RecordTime)
	RecordTime = 0;
end
TimeStamp = zeros(1,nImgs);

	% Create image output array 
	images = zeros(1336, 980, nImgs);

	% Open each image and read data
	for k = 1:nImgs
		fid = fopen(fpath(k));
		firstLine = fgetl(fid);

		% Define toggle for additional line read in header for new files with 4
		% header lines rather than the legacy of three
		
		if firstLine == "P5"
			additionalSkip = 1; 
		else 
			additionalSkip = 0; 
		end

		% If RecordTime, grab image time from header
		if RecordTime == 1
			Metadata = strsplit(fgetl(fid));
			Ind = find(contains(Metadata,'Time:'))+1;
			TimeStamp(k) = str2double(Metadata(Ind));
		else
    		fgetl(fid); 
		end	
		fgetl(fid);


		if additionalSkip == 1
			fgetl(fid);
		end

		% After skipping the starting three rows, grab image dat
		current_image = fread(fid,[1336,980],'uint16');
        fclose(fid);
	
		% Write image to output array 
		images(:,:,k) = current_image;
	end
end