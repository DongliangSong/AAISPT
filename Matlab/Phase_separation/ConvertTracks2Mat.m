clc;   clear

% files = dir('E:\Data\QiPan\Fig 4\Fig4a\*.csv');
files = 'E:\Data\QiPan\Fig 4\Fig4b\super small vacuole_9_Tracks.csv';
convertTrack2mat(files)

% for i = 1:length(files)
%     file = [files(i).folder,'\',files(i).name];
%     convertTrack2mat(file)
% end


function convertTrack2mat(path)

filename = strsplit(path, '.');
data = xlsread(path);
ind = data(:, 1);
unique_values = unique(ind);
first_indices = zeros(size(unique_values));

for i = 1:length(unique_values)
    first_indices(i) = find(ind == unique_values(i), 1);
end

if length(unique_values) == 1
    segs = data(:, 3:4);
else
    segs = {};
    for i = 1:length(first_indices)-1
        segs{i} = data(first_indices(i,1):first_indices(i+1,1)-1, 3:4);
    end
    segs{end+1} = data(first_indices(end,1):end, 3:4);
end

save([filename{1,1}, '.mat'], "segs");
end
