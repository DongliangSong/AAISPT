clc;   clear

% % for diffusion files
% ind = [002,003,010,040,043,044,047,045,046,049,053,055,067,068,069,073,074,075,078,082,084,102,103,091,090,106,113,112,105,124,133,135,130,141,140,150,158,154,160,159,172,171,170,175,177,176,182,186,193,201,189,205,206,210,211,212,217,226,277,384,403,406,407,408,498,507,508,509,510,511,512,502,503,504,505,506,514,549,546,538,534,535,331,332,039,076,032];

% % for other files 
% ind = [188, 200, 218, 024, 028, 030, 251];

% % for circling files
% ind = [005, 004, 092];

% for rocking files
ind = [214, 238, 469];
path = dir(fullfile('E:\轨迹分析\1\xyzap','*.xlsx'));

for i = 1:length(ind)
    index = num2str(ind(i),'%03d');
    
    for j = 1:length(path)
        in(j) = contains(path(j).name(1:3),index);
    end

    name = fullfile(path(in==1).folder,path(in==1).name);
    Table = readtable(name);
    Vars_name = Table.Properties.VariableNames;
    data = table2array(Table);
    azimuth = data(:,8);
    azimuth = Correct_angle(azimuth);
    data(:,8) = azimuth;
%     Vars_name(1,end-1:end) = {'Azimuth','Polar'};
    T = array2table(data,'VariableNames',Vars_name);
    writetable(T, name);
end


