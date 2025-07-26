clc;  clear

savepath = 'D:\TrajSeg-Cls\Exp Data\QiPan\Fig2\S1_Tracks\Tracks';
path = dir(fullfile(savepath, '*.xlsx*'));

for i = 1:length(path)
    filename = fullfile(path(i).folder,path(i).name);
    data = table2array(readtable(filename));
    data(:,2:3) = data(:,2:3) * 1000;     % Convert to nm unit.
    data = array2table(data);
    data.Properties.VariableNames = {'Frame','x','y'};
    [filepath, name, extend] = fileparts(filename);
    writetable(data,fullfile(savepath,[name,'.xlsx']));
end