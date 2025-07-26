clc;  clear

root = 'D:\TrajSeg-Cls\Exp Data\YanYu\Results\data';
% Find all .csv files in the target folder and all its subfolders
csvFiles = dir(fullfile(root, '**', '*.csv'));

for k = 1:length(csvFiles)
    csvFilePath = fullfile(csvFiles(k).folder, csvFiles(k).name);

    data = readtable(csvFilePath);
    [~, name, ~] = fileparts(csvFilePath);
    xlsxFilePath = fullfile(csvFiles(k).folder, [name, '.xlsx']);

    writetable(data, xlsxFilePath);
    fprintf('Converted %s to %s\n', csvFilePath, xlsxFilePath);
end

