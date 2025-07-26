%% Extract trajectory coordinates from .csv file.
clc;  clear

% nums = length(segs);
% for i=1:nums
%     data = segs{1,i};
%     plot(data(:,1),data(:,2))
%     axis equal 
%     I = getframe(gcf);
%     imwrite(I.cdata, ['E:\Data\QiPan\Fig2\S2_Tracks\img\', num2str(i),'.png']);
% end


path = 'D:\TrajSeg-Cls\Exp Data\QiPan\Fig2\S1_Tracks\S1_Tracks.csv';
folder = fileparts(path);

data = table2array(readtable(path));
ind = data(:,1);
unique_values = unique(ind);

first_indices = zeros(size(unique_values));

for i = 1:length(unique_values)
    first_indices(i) = find(ind == unique_values(i), 1);
end

for i = 1:length(first_indices)-1
    segs(:,2:3) = data(first_indices(i,1):first_indices(i+1,1)-1,3:4);
    segs(:,1) = 1:size(segs,1);
    writematrix(segs,fullfile(folder,[num2str(i) '.csv']));
    clear segs;
end

segs(:,2:3) = data(first_indices(end,1):end,3:4);
segs(:,1) = 1:size(segs,1);
writematrix(segs,fullfile(folder,[num2str(length(first_indices)) '.csv']));


%% Save each trajectory file to .mat for trajectory analysis.
path = 'E:\Data\QiPan\Fig3\all trajectories of 9 AuNRs_1676';
savepath = path;

files = dir([path,'\*']);
files(1:2,:) = [];
num_file = length(files);

data = cell(1,num_file);
for i = 1:num_file
    data{1,i} = table2array(readtable(fullfile(files(i).folder,files(i).name)));
end

save(fullfile(savepath,'tracks.mat'),"data");
