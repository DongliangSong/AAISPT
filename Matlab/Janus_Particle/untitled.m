clc;  clear

path = dir(fullfile('D:\TrajSeg-Cls\Exp Data\YanYu\Results','*.csv'));

for i=  1:length(path)
    filename = fullfile(path(i).folder,path(i).name);
    [~,name,~] = fileparts(filename);
    T = table([],[],'VariableNames',{'Frame','delta_azimuth'});

    writetable(T,fullfile('D:\TrajSeg-Cls\Exp Data\YanYu\Delta_azimuth',[name,'_deltaAZI.xlsx']));
end