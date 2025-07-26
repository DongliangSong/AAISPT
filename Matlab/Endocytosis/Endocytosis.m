clc;  clear

path = 'D:\TrajSeg-Cls\endoysis';
savepath = fullfile(path, 'StageSeg');

if ~exist('savepath','dir')
    mkdir(savepath);
end


% Extract xyzap trajectory
recursiveExtract(path,savepath,'xyzap.xlsx',[]);

% Extract xyap trajectory
recursiveExtract(path,savepath,'xyap.xlsx',[]);

% Extract xyz trajectory
recursiveExtract(path,savepath,'xyz.xlsx',[]);

% Extract xy trajectory
recursiveExtract(path,savepath,'xy.xlsx',[]);