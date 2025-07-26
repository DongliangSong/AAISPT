clc;  clear

total_len = length(xyz);
t = (1:1:total_len) * 0.001;
t = t';
dis = sqrt((xyz(:,1) - xyz(1,1)).^2+ (xyz(:,2) - xyz(1,2)).^2+ (xyz(:,3) - xyz(1,3)).^2);

%%
% Reverse the logit
eps = 1e-8;
prob = min(max(prob, eps), 1 - eps);
logit = log(prob ./ (1 - prob));

% Temperature scaling
figure
for i = 1:20
    prob_T(i,:) = 1 ./ (1 + exp(-logit ./ (i * 0.4)));
    plot(prob_T(i,:))
    hold on
end
legend(arrayfun(@(i) sprintf('Line %d', i), 1:20, 'UniformOutput', false));

%%  Merge trajectories
loc = CPs;
threshold = 3000;  % Distance threshold between two adjacent switching points
N = length(loc);
filtered_points = loc(1);  % Retain the first point
last_kept = loc(1);

for i = 2:N
    if loc(i) - last_kept >= threshold
        filtered_points(end+1) = loc(i);
        last_kept = loc(i);  % Update the most recently retained point
    end
end
N = length(filtered_points) - 1;
cmap = hsv(N);


figure
plot(t, xyz(:,1));
hold on
plot(t, xyz(:,2));
plot(t, xyz(:,3));
plot(t, dis,'LineWidth', 2);
ylabel('Displacement(\mum)','FontSize', 16, 'FontWeight', 'bold');
xlabel('Time(s)', 'FontSize', 16, 'FontWeight', 'bold');
hold on;
for i = 1:N
    idx1 = filtered_points(i);    
    idx2 = filtered_points(i+1);
    x1 = t(idx1);  
    x2 = t(idx2);
    xline(x1);
end
hold off;

% Set font and global properties
set(gca, 'FontName', 'Arial', 'FontSize', 14, 'LineWidth', 2);
set(gcf, 'Color', 'w');

%% Perform fitting to obtain D and alpha
cp = filtered_points;
seg = {};
seg{1} = xyz(1:cp(1), :);
for i = 1:length(cp) - 1
    seg{end + 1} = xyz(cp(i) + 1:cp(i + 1), :);
end
seg{end + 1} = xyz(cp(end) + 1:end, :);

clear D alpha
% Loop over segments
for i = 1:length(seg)
    trace = seg{i};
    nData = size(trace,1); 
    numberOfDeltaT = floor(nData/5); % for MSD, dt should be up to 1/5 of number of data points
    msd = zeros(numberOfDeltaT,3); 
    
    % calculate msd for all deltaT's
    for dt = 1:numberOfDeltaT
       deltaCoords = trace(1+dt:end,:) - trace(1:end-dt,:);
       squaredDisplacement = sum(deltaCoords.^2,2); % dx^2+dy^2+dz^2
    
       msd(dt,1) = mean(squaredDisplacement); % average
       msd(dt,2) = std(squaredDisplacement); % std
       msd(dt,3) = length(squaredDisplacement); % number of points
    end
    
    time = (1:length(msd(:,1)))/1000;
    
    % Define fitting function for MSD = 6*D*t^alpha
    fitfunc = @(p,t) 6*p(1)*t.^p(2); % p(1) = D, p(2) = alpha
    
    % Initial guess for [D, alpha]
    initialGuess = [1, 1]; % Starting with D=1, alpha=1
    
    % Perform curve fitting
    [p, ~] = lsqcurvefit(fitfunc, initialGuess, time', msd(:,1));
    D(i) = p(1); 
    alpha(i) = p(2);

%     % Plot MSD curve and fitted curve
%     figure; % Create a new figure for each segment
%     plot(time, msd(:,1), 'bo-', 'DisplayName', 'MSD Data'); % Plot MSD data
%     hold on;
%     % Generate fitted curve using fitted parameters
%     fitted_msd = fitfunc(p, time);
%     plot(time, fitted_msd, 'r-', 'DisplayName', sprintf('Fit: MSD = 6*%.4f*t^{%.4f}', p(1), p(2)));
%     hold off;
%     
%     % Add labels, title, and legend
%     xlabel('Time (s)');
%     ylabel('MSD (\mum^2)');
%     title(sprintf('MSD vs Time for Segment %d', i));
%     legend('show');
%     grid on;
end

clear new_alpha  new_D
new_D = [];  new_alpha = [];
for i = 1:length(seg)
    num = length(seg{i});
    new_D = [new_D; repmat(D(i),num,1)];
    new_alpha= [new_alpha; repmat(alpha(i),num,1)];
end

%% Plot
eff_len = length(new_alpha);
time = t(total_len - eff_len + 1:end);
time = time';

cp = [1, cp, 70688];  % for 3A-D

% Plot dual Y-axes: D and displacement
figure
yyaxis right
plot(time, dis,'LineWidth', 2);
ylabel('Displacement(\mum)','FontSize', 16, 'FontWeight', 'bold');
xlabel('Time(s)', 'FontSize', 16, 'FontWeight', 'bold');
yyaxis left
plot(time, new_D,'LineWidth', 2);
ylabel('D(\mum^2/s)', 'FontSize', 16, 'FontWeight', 'bold');

N = length(cp) - 1;
cmap = turbo(N);
hold on;
for i = 1:N
    idx1 = cp(i);    
    idx2 = cp(i+1);
    x1 = t(idx1);  
    x2 = t(idx2);
    y_min = 0;   
    y_max = D(i); 
    rectangle('Position', [x1, y_min, x2 - x1, y_max - y_min], ...
              'FaceColor', cmap(i,:), 'EdgeColor', 'none');
end
hold off;

% Set font and global properties
set(gca, 'FontName', 'Arial', 'FontSize', 14, 'LineWidth', 2);
set(gcf, 'Color', 'w');

%% 
% Plot dual Y-axes: alpha vs displacement
figure
yyaxis right
plot(time, dis,'LineWidth', 2);
ylabel('Displacement(\mum)','FontSize', 16, 'FontWeight', 'bold');
xlabel('Time(s)', 'FontSize', 16, 'FontWeight', 'bold');
yyaxis left
plot(time, new_alpha, 'LineWidth', 2);
ylabel('alpha', 'FontSize', 16, 'FontWeight', 'bold');

N = length(cp) - 1;
cmap = turbo(N);
hold on;
for i = 1:N
    idx1 = cp(i);    
    idx2 = cp(i+1);
    x1 = t(idx1);  
    x2 = t(idx2);
    y_min = 0;   
    y_max = alpha(i); 
    rectangle('Position', [x1, y_min, x2 - x1, y_max - y_min], ...
              'FaceColor', cmap(i,:), 'EdgeColor', 'none');
end
hold off;

% Set font and global properties
set(gca, 'FontName', 'Arial', 'FontSize', 14, 'LineWidth', 2);
set(gcf, 'Color', 'w');

%% Plot 3D trajectory
colors = turbo(length(cp));

lineWidth = 1;
fontSize = 16;
fontName = 'Arial';
fontWeight = 'bold';
labelFontSize = 16;
labelFontName = 'Arial';
labelFontWeight = 'bold';
axisLineWidth = 2;

figure
hold on;
for i = 1:length(cp) - 1
    plot3(xyz(cp(i):cp(i + 1) , 1), xyz(cp(i):cp(i + 1), 2), ...
        xyz(cp(i):cp(i + 1), 3), 'Color', colors(i,:),'LineWidth', lineWidth);
end
hold off
set(gca, 'FontSize', fontSize, 'FontName', fontName, 'LineWidth', axisLineWidth); 
axis equal;
xlabel('X(\mum)','FontSize', 16, 'FontWeight', 'bold');
ylabel('Y(\mum)','FontSize', 16, 'FontWeight', 'bold');
zlabel('Z(\mum)','FontSize', 16, 'FontWeight', 'bold');
colorbar;
colormap(colors);
caxis([1 length(cp)])
axis tight;
view(3);

%% Plot 3D trajectory, colored by diffusion coefficient
% Create colormap mapping
cmap = turbo(256); 
cmin = min(D);
cmax = max(D);

% Map D to color index
color_idx = round(1 + (D - cmin) / (cmax - cmin) * (size(cmap,1) - 1));
color_idx = max(min(color_idx, size(cmap,1)), 1);

figure
hold on;
for i = 1:length(cp) - 1
    plot3(xyz(cp(i):cp(i + 1) , 1), xyz(cp(i):cp(i + 1), 2), ...
        xyz(cp(i):cp(i + 1), 3), 'Color', cmap(color_idx(i),:),'LineWidth', lineWidth);
end
hold off
set(gca, 'FontSize', fontSize, 'FontName', fontName, 'LineWidth', axisLineWidth); 
axis equal;
xlabel('X(\mum)','FontSize', 16, 'FontWeight', 'bold');
ylabel('Y(\mum)','FontSize', 16, 'FontWeight', 'bold');
zlabel('Z(\mum)','FontSize', 16, 'FontWeight', 'bold');

colormap(cmap);
cb = colorbar;
cb.Label.String = 'D (\mum^2/s)';
caxis([cmin, cmax]);
axis tight;
view(3);

%%
figure
plot3(seg{1, 2}(:,1),seg{1, 2}(:,2),seg{1, 2}(:,3),'r');
hold on
plot3(seg{1, 3}(:,1),seg{1, 3}(:,2),seg{1, 3}(:,3),'g');
plot3(seg{1, 4}(:,1),seg{1, 4}(:,2),seg{1, 4}(:,3),'b');
axis equal;
hold off;

figure
plot(dis)

figure
plot(seg{1, 4}(:,1));
hold on
plot(seg{1, 4}(:,2));
plot(seg{1, 4}(:,3));



