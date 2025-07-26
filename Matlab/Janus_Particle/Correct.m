clc;   clear

path = 'D:\TrajSeg-Cls\Exp Data\YanYu\Results\data\diffusion\003-050717_BW1_T14_T1_New_thresh_results.xlsx';

data = table2array(readtable(path));
azimuth = data(:,8);
dif = abs(diff(azimuth));

new = azimuth;
while max(dif) > 180
    for i = 1:length(dif)
        if dif(i) > 180
            if azimuth(i) < azimuth(i+1)
                azimuth(i+1) = azimuth(i+1) - 360;
            else
                azimuth(i+1) = azimuth(i+1) + 360;
            end
            dif = abs(diff(azimuth));
        end
    end
end