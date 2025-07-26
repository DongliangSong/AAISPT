clc; clear

load('E:\Data\QiPan\Fig2\S1-S2_Tracks\S1-S2track.mat');
nums = length(segs);

ind = [];
for i = 1:nums
    data = segs{1,i};

    index1 = find(data(:,1)~=0,1);
    index2 = find(data(:,2)~=0,1);
    index = max(index1,index2);
    data = data(index:end,:);

    if all(data(:,1)==0) || all(data(:,2)==0)
        ind = [ind,i];
    elseif all(data(:,1)==data(1,1)) || all(data(:,2)==data(1,2))
        ind = [ind,i];
%     elseif abs(max(data(:,1))-min(data(:,1))) < 1.5 || abs(max(data(:,2))-min(data(:,2))) <1.5
%         ind = [ind,i];
    end
end

for i = 1:length(ind)
    segs(ind(i)) = [];
end