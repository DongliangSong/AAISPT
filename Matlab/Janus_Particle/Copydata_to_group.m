clc;
clear;

rootpath = 'D:\TrajSeg-Cls\Exp Data\YanYu\Results\data';
categoryFolders = {'Circling', 'Confined_circling', 'Confined', 'Rocking', ...
                  'Diffusion', 'None', 'Other'};

% Create target folders
for i = 1:length(categoryFolders)
    folderPath = fullfile(rootPath, categoryFolders{i});
    if ~exist(folderPath, 'dir')
        mkdir(folderPath);
        fprintf('Created folder: %s at %s\n', folderPath, datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    end
end

% Load data and label
dataRoot = 'D:\TrajSeg-Cls\Exp Data\YanYu\Results';
csvFiles = dir(fullfile(dataRoot,'*.csv'));
label = readtable(fullfile(dataRoot,'label','label_config.csv'));

% Define the mapping between labels and folders
labelMap = containers.Map(...
    {1, 2, 3, 4, 5, 6, 7}, ...
    categoryFolders);

% Process each CSV file
numFiles = length(csvFiles);
for i = 1:numFiles
    % Get the filename and full path
    filePath = fullfile(csvFiles(i).folder, csvFiles(i).name);
    [~, fileName, fileExt] = fileparts(filePath);

    % Find the corresponding label
    matchIdx = strcmp(label.Name, fileName);
    if sum(matchIdx) == 1
        labelIdx = label.Label(matchIdx);
        if isKey(labelMap, labelIdx)
            targetFolder = fullfile(rootPath, labelMap(labelIdx));
            
            % Copy the file to the target folder
            copyfile(filePath, fullfile(targetFolder, [fileName, fileExt]));
            fprintf('Copied %s to %s at %s\n', [fileName, fileExt], targetFolder, datestr(now, 'yyyy-mm-dd HH:MM:SS'));
        else
            warning('No mapping found for label %d in file %s', labelIdx, fileName);
        end
    else
        warning('No unique match found for file %s in label table', fileName);
    end
end





