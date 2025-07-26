% Optimization algorithm for filling window length
% Used for Supporting information

clc;  clear

load('xy.mat');

data = xy;
data_copy = data;
min_error = Inf;
best_window_length = [];
temp = data(~ismissing(data(:,1)),:);
gt = temp;

minlen = 5;         
maxlen = floor(length(temp) * 0.05);         
randlen = randi([minlen,maxlen]);

maxstart = length(temp) - randlen + 1;
randstart = randi([1, maxstart]);
temp(randstart:(randstart + randlen - 1),:) = NaN;


errors = [];
for wl = 40:10:700   % window length (The minimum window length should be greater than the missing value length.)
    filled = fillmissing(temp,'movmedian',wl);  

    error = mean((gt - filled).^2);
    errors = [errors;error];
    
    % Update minimum error and optimum window length
    if mean(error) < min_error
        min_error = mean(error);
        best_window_length = wl;
    end
end

% Fill using optimal window length
data_filled = fillmissing(data, 'movmedian', best_window_length);

