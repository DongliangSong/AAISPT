clc;   clear

% Data augmentation by dimensional segmentation.
savepath = 'D:\TrajSeg-Cls\endoysis\Enhanced';
if ~exist("savepath","dir")
    mkdir(savepath)
end

path = dir('D:\TrajSeg-Cls\endoysis\StageSeg1');
path(1:2,:) = [];

nums = length(path);
for i = 1:nums
    filename = fullfile(path(i).folder,path(i).name);
    data = readtable(filename);

    [filepath,name,ext] = fileparts(filename);
    suffix = strsplit(name,'_');

    if strcmp(suffix{end}, 'xyzap')
        data_xy = data(:,1:4);
        data_xyz = data(:,1:5);
        data_xyap = [data(:,1:4) data(:,6:7)];
        data_xyzap = data;

        writetable(data_xy, fullfile(savepath,[name,'_f_xy.xlsx']));
        writetable(data_xyz, fullfile(savepath,[name,'_f_xyz.xlsx']));
        writetable(data_xyap, fullfile(savepath,[name,'_f_xyap.xlsx']));
        writetable(data_xyzap, fullfile(savepath,[name,'_f_xyzap.xlsx']));
            
    elseif strcmp(suffix{end}, 'xyap')
        data_xy = data(:,1:4);
        data_xyap = data;

        writetable(data_xy, fullfile(savepath,[name,'_f_xy.xlsx']));
        writetable(data_xyap, fullfile(savepath,[name,'_f_xyap.xlsx']));

    elseif strcmp(suffix{end}, 'xyz')
        data_xy = data(:,1:4);
        data_xyz = data;

        writetable(data_xy, fullfile(savepath,[name,'_f_xy.xlsx']));
        writetable(data_xyz, fullfile(savepath,[name,'_f_xyz.xlsx']));

    elseif strcmp(suffix{end}, 'xy')
        data_xy = data;
        writetable(data_xy, fullfile(savepath,[name,'_f_xy.xlsx']));
    end
end






