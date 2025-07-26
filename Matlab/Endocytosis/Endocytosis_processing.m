clc;   clear

% Deletes empty columns in the data table and fills in the missing values with linear interpolation.

% Load target folders
path = dir('D:\TrajSeg-Cls\endoysis\Enhanced\*.xlsx');


path = path([path.isdir]);

nums = length(path);
for i = 1:nums
    files = dir(fullfile(path(i).folder,path(i).name,'*.xlsx'));

    num_file = length(files);
    for j = 1:num_file
        filename = fullfile(files(j).folder,files(j).name);
        [filepath,name,ext] = fileparts(filename);

        if strcmp(name,'xy') || strcmp(name,'xyz') || strcmp(name,'xyap') || strcmp(name,'xyzap') || strcmp(name,'peru_xy')
            data = readtable(filename);
            
            if ~strcmp(data.Properties.VariableNames{1},'t')
                newcol = table2array(data(:,1)) * 0.02;

                % Add a time column
                newcolname = 't';
                newcoltable = table(newcol,'VariableNames', {newcolname});
                Table = [newcoltable data];
            else
                Table = data;
            end

            % Delete empty columns
            Table(:, all(ismissing(Table))) = [];

            % Check and fill in missing values in each column
            for k = 1:width(Table)
                if any(ismissing(Table{:,k}))
                    % If there are missing values in the columns, use interpolation to fill them in.
                    Table{:,k} = fillmissing(Table{:,k}, 'linear');
                end
            end

            writetable(Table, filename);
        end
    end
end

