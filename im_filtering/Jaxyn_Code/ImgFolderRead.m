function [OutFile,nOutFiles] = ImgFolderRead(folder, filetype, filenumber)
%Description - Takes in a directory of files of the same type and returns a matrix of the
%files containted within the directory
%
% Inputs:
%    folder         - Directory
%    filetype       - Type of files contained in the directory (Must be same file type) 
%    filenumber     - File(s) to be returned, 'all' if entire directory
%   
% Outputs:
%    OutFile        - Matrix of files 
%
% Other im-files required: none
% Subfunctions: none
%
%
% Author: Thomas Metcalfe
% July 16, 2020
%
% Modified:
% 
%
%------------- BEGIN CODE --------------%  

Files = dir(fullfile(folder,filetype));
numfiles = length(Files);
nOutFiles = 0;

if length(filenumber) == 1 && ischar(filenumber) == 0     %Case of being passed a single file
    first = filenumber;
    last = filenumber;
    filesize = 1;
    fileflag = 0;
elseif length(filenumber) > 1 && ischar(filenumber) == 0  %Case of being passed a range of files
    first = filenumber(1);
    last = filenumber(end);
    filesize = last - first + 1;
    fileflag = 0;
elseif ischar(filenumber) == 1                            %Case of being passed the entire folder
    first = 1;
    last = numfiles;
    fileflag = 0;
    filesize = numfiles;
else
    fileflag = 1;
end


if fileflag == 0
    FileHist = strings(filesize,1);
    ii = 1;
    for i = first:last
        CurrentFile = Files(i);
        FileName = CurrentFile.name;
        FileHist(ii) = FileName;
        nOutFiles = nOutFiles + 1;
        ii = ii + 1;
    end
    OutFile = FileHist;
else
    OutFile = 'Unsupported number of files';
    
 
end

