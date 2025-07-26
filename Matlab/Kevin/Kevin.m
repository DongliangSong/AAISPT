clc;  clear

root = 'E:\Data\Kevin\Fig 2F-K TrIm data\Processed Files\210430 VID0103 TR0103 Fig2F_K\16 FPV';
openfig(fullfile(root, 'Trajectory.fig'));
fig = gcf;
ax = gca;

% Obtain graph data
line_handles = get(ax, 'Children');
x_data = get(line_handles, 'XData');
y_data = get(line_handles, 'YData');
z_data = get(line_handles, 'ZData');
writematrix(y_data', fullfile(root, 'Trajectory.csv'));

xx = [];
yy = [];
zz = [];

for i = 1:length(x_data)
    x = x_data{i};
    y = y_data{i};
    z = z_data{i};
    xx = [xx; x'];
    yy = [yy; y'];
    zz = [zz; z'];
%     plot3(x,y,z)
%     hold on
end
axis equal
writematrix([xx,yy,zz], fullfile(root,'Trajectory.csv'));

path = fullfile(root,'TopHatFiltChangePointFigs');
for i = 1:length(traj)
    a = [traj(i).x,traj(i).y,traj(i).z];
    writematrix(a,fullfile(path,[num2str(i) '.csv']));
end

