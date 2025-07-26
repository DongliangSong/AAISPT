clc;  clear

path = 'D:\TrajSeg-Cls\endoysis';
savepath = fullfile(path, 'StageSeg');

if ~exist('savepath','dir')
    mkdir(savepath);
end


% 提取xyzap轨迹
recursiveExtract(path,savepath,'xyzap.xlsx',[]);

% 提取xyap轨迹
recursiveExtract(path,savepath,'xyap.xlsx',[]);

% 提取xyz轨迹
recursiveExtract(path,savepath,'xyz.xlsx',[]);

% 提取xy轨迹
recursiveExtract(path,savepath,'xy.xlsx',[]);