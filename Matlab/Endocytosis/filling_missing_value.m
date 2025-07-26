clc;  clear

% The missing values in the data are processed using linear interpolation.
path = dir('D:\TrajSeg-Cls\endoysis\Enhanced\*.xlsx');

for i = 1:length(path)
    filename = fullfile(path(i).folder,path(i).name);
    Table = readtable(filename);

    % delete null column
    Table(:, all(ismissing(Table))) = [];

    % Check and fill in missing values in each column
    for k = 1:width(Table)
        if any(ismissing(Table{:,k})) 
            Table{:,k} = fillmissing(Table{:,k}, 'movmean',200);  
        end
    end

    writetable(Table, filename);
end




























