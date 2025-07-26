% Filling

clc;   clear;

path = dir('D:\TrajSeg-Cls\endoysis\Enhanced\*.xlsx');
savepath = 'D:\TrajSeg-Cls\endoysis\Enhanced\Filling_processing';

if ~exist("savepath",'dir')
    mkdir(savepath);
end

for n = 1:length(path)
    filename = fullfile(path(n).folder, path(n).name);
    [~, name, ext] = fileparts(filename);

    T = readtable(filename);
    Vars_name = T.Properties.VariableNames;
    T = table2array(T);
    T(T == 0) = NaN;

    for m = 1:size(T, 2)
        data = T(:, m);

        % Identify all NaN segments
        [nanStart, nanEnd, nanLengths] = findNaNSequences(data);

        % Initialize the result array
        filledArray = data;

        % Define different window lengths
        windowLengths = nanLengths + 10;

        % Check if the number of NaN segments matches the number of window lengths
        if length(nanStart) ~= length(windowLengths)
            error('The number of NaN segments does not match the number of window lengths');
        end

        % Apply moving average filling to each NaN segment using different window lengths
        for k = 1:length(nanStart)
            startIdx = nanStart(k);
            endIdx = nanEnd(k);
            windowLength = windowLengths(k);

            % Extract NaN segments
            subArrayStart = max(1, startIdx - floor(windowLength / 2));
            subArrayEnd = min(length(data), endIdx + floor(windowLength / 2));
            subArray = data(subArrayStart:subArrayEnd);

            % Fill NaN using the moving average method
            subArray = fillmissing(subArray, 'movmean', windowLength);

            % Place the filled data back into the result array
            filledArray(startIdx:endIdx) = subArray((startIdx - subArrayStart + 1):(endIdx - subArrayStart + 1));
        end

        T(:, m) = filledArray;
    end

    new_table = array2table(T,'VariableNames',Vars_name);
    writetable(new_table,fullfile(savepath,[name,ext]));
end

function [nanStart, nanEnd, nanLengths] = findNaNSequences(data)
    nanStart = [];
    nanEnd = [];
    nanLengths = [];
    currentNaNCount = 0;

    for i = 1:length(data)
        if isnan(data(i))
            if currentNaNCount == 0
                nanStart(end + 1) = i;  % Record the start position of NaN segments
            end
            currentNaNCount = currentNaNCount + 1;
        else
            if currentNaNCount > 0
                nanEnd(end + 1) = i - 1;  % Record the end position of NaN segments
                nanLengths(end + 1) = currentNaNCount;
                currentNaNCount = 0;
            end
        end
    end

    % Check if there are consecutive NaNs at the end of the array
    if currentNaNCount > 0
        nanEnd(end + 1) = length(data);
        nanLengths(end + 1) = currentNaNCount;
    end
end
