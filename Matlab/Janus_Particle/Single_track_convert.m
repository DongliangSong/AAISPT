clc;  clear

path = dir(fullfile('D:\TrajSeg-Cls\Exp Data\YanYu\Results\data\diffusion','*.xlsx'));

for i = 1:length(path)
    filename = fullfile(path(i).folder,path(i).name);
    if contains(filename,'Single_track')
        table = readtable(filename);
        Vars_name = table.Properties.VariableNames;
        data = table.Variables;
        data(:,2:3) = data(:,2:3) * 1000;
        T = array2table(data,'VariableNames',Vars_name);
        writetable(T,filename);
    end
end