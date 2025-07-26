clc;   clear

path = 'D:\TrajSeg-Cls\Exp Data\YanYu';
dirs = dir([path,'\*']);
dirs = dirs(~ismember({dirs.name}, {'.', '..'}));

num_dir = length(dirs);
for i = 1:num_dir
    files = dir(fullfile(dirs(i).folder, dirs(i).name));
    files = files(~ismember({files.name}, {'.', '..'}));
    num_file = length(files);
    
    for j = 1:num_file
        filename = fullfile(files(j).folder,files(j).name);
        [folder,name,ext] = fileparts(filename);
        parts = strsplit(folder,'\');

        if strcmp(ext,'.opj') || strcmp(ext,'.opju')
            continue
        end

        % load data
        data = table2array(readtable(filename));
        num = size(data,1);
        time = data(:,end-1);

        if all(data(:,3) == 0) || all(data(:,4) == 0)
            disp(filename);
        end
    end
end
