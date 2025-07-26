clc;  clear

root = 'D:\TrajSeg-Cls\Exp Data\YanYu\Results';
path = dir(fullfile(root,'*.csv'));

savepath = fullfile(root, 'img2');
if ~exist('savepath','dir')
    mkdir(savepath);
end

% plot traces
for i = 1:length(path)
    filename = fullfile(path(i).folder, path(i).name);
    data = table2array(readtable(filename));
    plot(data(:, 2), data(:, 3));

    if size(data,2) > 3
        hold on
        plot(data(:, 4),data(:, 5))
    end
    hold off
    axis equal
    [~, name, ~] = fileparts(filename);
    title(name)
    I = getframe(gcf);
    imwrite(I.cdata, fullfile(savepath, [num2str(i,'%03d'),'.png']));
end

