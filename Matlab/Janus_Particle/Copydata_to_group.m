clc;   clear

path = 'D:\TrajSeg-Cls\Exp Data\YanYu\Results\data';
Circling = fullfile(path, 'circling');
Confined_circling = fullfile(path,'confined_circling');
Confined = fullfile(path,'confined');
Rocking = fullfile(path,'rocking');
Diffusion = fullfile(path,'diffusion');
None = fullfile(path,'none');
Other = fullfile(path,'other');

if ~exist("Circling",'dir');   mkdir(Circling);  end
if ~exist("Confined_circling",'dir');  mkdir(Confined_circling);  end
if ~exist("Confined",'dir');   mkdir(Confined);  end
if ~exist("Rocking",'dir');  mkdir(Rocking);  end
if ~exist("Diffusion",'dir');  mkdir(Diffusion);  end
if ~exist("None",'dir');   mkdir(None);   end
if ~exist("Other",'dir');   mkdir(Other);   end

% Load data and label
root = dir(fullfile('D:\TrajSeg-Cls\Exp Data\YanYu\Results','*.csv'));

label = readtable('D:\TrajSeg-Cls\Exp Data\YanYu\Results\label\label_config.csv');

nums = length(root);
for i = 1:nums
    filename = fullfile(root(i).folder,root(i).name);
    [~, name, ext] = fileparts(filename);

    IndexC = strfind(label.Name,name);
    Index = find(~(cellfun('isempty', IndexC))); 
    
    ind = label.Label(Index);
    % 1 for 'Circling'
    % 2 for 'Confined'
    % 3 'Confined_circling'
    % 4 'Diffusion'
    % 5 'None'
    % 6 'Other'
    % 7 'Rocking'

    if ind==1
        target = Circling;
    elseif ind==2
        target = Confined;
    elseif ind == 3
        target = Confined_circling;
    elseif ind == 4
        target = Diffusion;
    elseif ind == 5
        target = None;
    elseif ind == 6
        target = Other;
    elseif ind == 7
        target = Rocking;
    end
    
    % Copy source data to target group
    copyfile(filename,fullfile(target,[name,ext]));
end





