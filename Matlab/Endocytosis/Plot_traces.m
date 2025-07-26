clc;   clear

path = dir(fullfile('D:\TrajSeg-Cls\endoysis\Enhanced','*.xlsx'));

savepath = 'D:\TrajSeg-Cls\endoysis\Enhanced\img';

if ~exist('savepath','dir')
    mkdir(savepath);
end

for i = 1:length(path)
    filename = fullfile(path(i).folder,path(i).name);
    [~,name,~] = fileparts(filename);

    data = table2array(readtable(filename));
    xy = data(:,3:4);
    plot(xy(:,1),xy(:,2));
    axis equal
    title(name)

    I = getframe(gcf);
    imwrite(I.cdata, fullfile(savepath, [num2str(i,'%03d'),'.png']));
end


clc;  clear

path = dir(fullfile('D:\TrajSeg-Cls\endoysis\Enhanced','*.xlsx'));
savepath = 'D:\TrajSeg-Cls\endoysis\Enhanced\apimg';

if ~exist('savepath','dir')
    mkdir(savepath);
end


for i = 1:length(path)
    filename = fullfile(path(i).folder,path(i).name);
    [~,name,~] = fileparts(filename);
    a = strsplit(name,'_');

    if contains(a{end},'ap')
        data = table2array(readtable(filename));

        ap = data(:,end-1:end);

        plot(ap(:,1))
        
        title(name)

        I = getframe(gcf);
        imwrite(I.cdata, fullfile(savepath, [name(1:3),'.png']));
    end
end

