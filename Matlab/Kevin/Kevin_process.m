clc;   clear

data = {};
folderPath = 'D:\TrajSeg-Cls\Exp Data\Kevin';
data = recursiveExtract(folderPath, data);


function data = recursiveExtract(target_folder, data)
% Recursive method to extract files of the specified type from a folder.

files = dir(target_folder);

for i = 1:length(files)
    if strcmp(files(i).name, '.') || strcmp(files(i).name, '..')
        continue;
    end

    fullPath = fullfile(target_folder, files(i).name);

    if files(i).isdir
        data = recursiveExtract(fullPath, data);

    elseif strcmp(files(i).name(end-3:end), '.csv')
        csvData = readtable(fullPath);
        data{end+1} = csvData;
    end
end
end