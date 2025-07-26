clc;   clear

% 基于Stepping-find HMM方法分割粒子轨迹
% 对sfHMM算法处理结果进行后处理

mainpname = uigetdir; %Select the folder which contains .xlsx
xyzap = xlsread([mainpname,'\xyzap.csv']);
xyz = xyzap(:,1:3);
dis = sqrt(((xyz(:,1)-xyz(1,1)).^2 + (xyz(:,2)-xyz(1,2)).^2) + (xyz(:,3)-xyz(1,3)).^2));


% 导入sfHMM分割结果，进行处理
D = importdata([mainpname,'\sfHMM_dis.txt']);
segD = D.data;
mD = size(segD,1);
cpD = change_point_detection(segD,thresholds);


% Set parameters   ************
thresholds = 200;  % 
diff_cp = 50;    % 相邻切换点之间的距离阈值


% merge 
cp = unique(cpD);

final_cp = [];
i = 1;
while i <= length(cp)
    j = i + 1;
    while j <= length(cp) && cp(j) - cp(i) <= diff_cp   % 相邻索引小于20则求均值代替
        j = j + 1;
    end
    
    final_cp = [final_cp, floor(mean(cp(i:j-1)))];
    i = j;
end

cp = final_cp;

% plot segmentated trajectory
color = {'r';'g';'b';'k';'y'; 'm'; 'c'};
colors = repmat(color,20,1);

% plot xy trajectory
figure
plot3(xyz(1:cp(1),1),xyz(1:cp(1),2),xyz(1:cp(1),3),'*')
hold on
for i = 1:length(cp)-1
    plot3(xyz(cp(i):cp(i+1),1),xyz(cp(i):cp(i+1),2),xyz(cp(i):cp(i+1),3),'Color',colors{i});
end
plot3(xyz(cp(end):mD,1),xyz(cp(end):mD,2),xyz(cp(end):mD,3),'Color',colors{length(cp)});
hold off
axis equal
title('XY trajectory')

% plot displacement
figure
plot(1:cp(1),dis(1:cp(1),1),'*')
hold on
for i = 1:length(cp)-1
    plot(cp(i):cp(i+1),dis(cp(i):cp(i+1),1),'Color',colors{i});
end
plot(cp(end):mA-1,dis(cp(end):end,1),'Color',colors{length(cp)});
hold off
axis equal
title('Displacement')


