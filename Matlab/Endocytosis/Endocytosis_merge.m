clc;   clear

% Merge data from each subfolder into the specified folder.

% Load target folders
path = dir('D:\TrajSeg-Cls\endoysis\*');
path(1:2,:) = [];
path = path([path.isdir]);

savepath = 'D:\TrajSeg-Cls\endoysis\StageSeg1';
if ~exist("savepath","dir")
    mkdir(savepath);
end

% load label config
config = readtable('D:\TrajSeg-Cls\endoysis\label_config.xlsx');
nums = size(config,1);

cellarray_label = config.folder;
for i = 1:length(path)
    cellarray_path{i,1} = path(i).name;
end

commonElements = intersect(cellarray_label, cellarray_path);

for i = 1:length(cellarray_label) 
    element = cellarray_label{i};
    if ismember(element, commonElements)
        index(i,1) = find(strcmp(cellarray_path, element));
    else
        index(i,1) = [];
    end
end


% load each stage time
stage1_start = config.stage1_start;
stage1_end = config.stage1_end;
stage2_start = config.stage2_start;
stage2_end = config.stage2_end;
stage3_start = config.stage3_start;
stage3_end = config.stage3_end;
stage4_start = config.stage4_start;
stage4_end = config.stage4_end;

for i = 1:nums
    ind = index(i,1);
    files = dir(fullfile(path(ind).folder,path(ind).name));
    files(1:2,:) = [];

    num_file = length(files);
    for j = 1:num_file
        filename = fullfile(files(j).folder,files(j).name);
        [filepath,name,ext] = fileparts(filename);

        if strcmp(name,'xy') || strcmp(name,'xyz') || strcmp(name,'xyap') || strcmp(name,'xyzap') 
            data = readtable(filename);
            t = data.t;

            if isnan(stage1_start(i)) || isnan(stage1_end(i))
                s1_range = [];
            else
                s1_range = find(t >= stage1_start(i) & t <= stage1_end(i));
            end

            if isnan(stage2_start(i)) || isnan(stage2_end(i))
                s2_range = [];
            else
                s2_range = find(t >= stage2_start(i) & t <= stage2_end(i));
            end

            if isnan(stage3_start(i)) || isnan(stage3_end(i))
                s3_range = [];
            else
                s3_range = find(t >= stage3_start(i) & t <= stage3_end(i));
            end

            if isnan(stage4_start(i)) || isnan(stage4_end(i))
                s4_range = [];
            else
                s4_range = find(t >= stage4_start(i) & t <= stage4_end(i));
            end

            stage1 = data(s1_range,:);
            stage2 = data(s2_range,:);
            stage3 = data(s3_range,:);
            stage4 = data(s4_range,:);

            [~,savename,~] = fileparts(filepath);

            stage_name = FileRenaming(savepath, savename, name, ext);
            writetable(stage1,stage_name{1});
            writetable(stage2,stage_name{2});
            writetable(stage3,stage_name{3});
            writetable(stage4,stage_name{4});
        end
    end
end

