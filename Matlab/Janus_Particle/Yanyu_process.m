clc;  clear

% load data
data = xlsread('E:\Data\YanYu\051717_S1\051717_S1_T1_NP1.csv');
loc = data(:,1:4);
time = data(:,5);
num = size(loc,1);

NP1_len = 200;    % large particle size.
NP2_len = 40;     % small particle size.
NP_len = (NP1_len + NP2_len)/2;    % radius
pixel_size = 106;     % pixel size

x1 = loc(:,1) * pixel_size;   % Actual position in nm unit.
y1 = loc(:,2) * pixel_size;
x2 = loc(:,3) * pixel_size;
y2 = loc(:,4) * pixel_size;

% Calculate the in-plane angle (phi) between the projection of the vector onto the xy-plane and the x-axis.
delta_x = x2 - x1;
delta_y = y2 - y1;
phi = atan2d(delta_y,delta_x);
phideg = atand(delta_y ./ delta_x);

% Calculate the out-plane angle (theta) between the vector and the z-axis.
projection_length = sqrt(delta_x.^2 + delta_y.^2);
theta = asind(projection_length ./ NP_len);

% Save results
ind = linspace(1,num,num);
Table = array2table([ind',phi,theta], 'VariableNames', {'Index','Azimuth', 'Polar'});

filename = fullfile('E:\Data\YanYu\051717_S1','results.csv');
writetable(Table, filename);


% Plot trace
figure
plot(loc(:,1),loc(:,2),'r','LineWidth',2)   % large particle
hold on
plot(loc(:,3),loc(:,4),'g','LineWidth',2)   % small particle
hold off
axis equal
xlabel('X (nm)')
ylabel('Y (nm)')
box off
set(gca,'LineWidth', 2,'FontSize', 14, 'FontWeight', 'bold')
saveas(gcf, 'plot_figure.png')

% Draw vectors between particles.
figure
for i = 1:size(loc, 1)
    x = [loc(i, 1), loc(i, 3)];
    y = [loc(i, 2), loc(i, 4)];
    plot(x, y, 'b','LineWidth',2);
    hold on;

    arrow_pos_x = loc(i, 3);
    arrow_pos_y = loc(i, 4);
    quiver(loc(i, 1), loc(i, 2), ...
           arrow_pos_x - loc(i, 1), arrow_pos_y - loc(i, 2), ...
           0, 'MaxHeadSize', 1);
end

axis equal;
xlabel('X (nm)');
ylabel('Y (nm)');
title('Vector direction');
grid on;
set(gca,'LineWidth', 2,'FontSize', 14, 'FontWeight', 'bold')
saveas(gcf, 'plot_figure.png');



