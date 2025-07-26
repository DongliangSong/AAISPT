% Convert the polar angle to the 0-90 range

clc;   clear

path = dir('D:\TrajSeg-Cls\endoysis\Enhanced\Filling_processing\*.xlsx');
savepath = 'D:\TrajSeg-Cls\endoysis\Enhanced\Transform';
if ~exist('savepath','dir')
    mkdir(savepath);
end

for i = 1:length(path)
    filename = fullfile(path(i).folder, path(i).name);
    [~, name, ext] = fileparts(filename);
    suffix = strsplit(name,'_');

    Table = readtable(filename);
    vars_name = Table.Properties.VariableNames;

    if strcmp(suffix{end}, 'xyzap') || strcmp(suffix{end}, 'xyap')
        data = table2array(Table);
        polar = data(:, end);
        polar(polar > 90) = 180 - polar(polar > 90);
        data(:, end) = polar;

        table = array2table(data,'VariableNames', vars_name);
        writetable(table, fullfile(savepath, [name, ext]));
    else
        writetable(Table, fullfile(savepath, [name, ext]));
    end
end