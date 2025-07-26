clc;  clear

% Create labels based on the groups divided in the img folder
path = dir(fullfile('D:\TrajSeg-Cls\Exp Data\YanYu\Results','*.csv'));
savepath = 'D:\TrajSeg-Cls\Exp Data\YanYu\Results\label';

nums = length(path);
names = {};
for i = 1:nums
    filename = fullfile(path(i).folder,path(i).name);
    [filepath,name,ext] = fileparts(filename);
    names{i,1} = name;
end

img_path = dir('D:\TrajSeg-Cls\Exp Data\YanYu\Results\img2');
img_path = img_path(~ismember({img_path.name}, {'.', '..'}));
% 1 for 'Circling'
% 2 for 'Confined'
% 3 'Confined_circling'
% 4 'Diffusion'
% 5 'None'
% 6 'Other'
% 7 'Rocking'

T = table([], [], [], 'VariableNames', {'Name', 'Index','Label'});
num_group = length(img_path);
for i = 1:num_group
    imageFiles = dir(fullfile(img_path(i).folder,img_path(i).name,'*.png'));

    for k = 1:length(imageFiles)
        [~, fileName, ~] = fileparts(imageFiles(k).name);
        name = names(str2double(fileName),1);
        Table = table(name, cellstr(fileName),i,'VariableNames', {'Name', 'Index','Label'});
        T = [T; Table];
    end
end

writetable(T,fullfile(savepath,'label_config.csv'));


