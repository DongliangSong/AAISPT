function stage_name = FileRenaming(savepath, savename, name, ext)
% 

stages = {'stage1', 'stage2', 'stage3', 'stage4'};

for i = 1:length(stages)
    stage_name{i} = fullfile(savepath, [savename, '_', stages{i}, '_', name, ext]);

    % Rename the file if it exists.
    if isfile(stage_name{i})
        stage_name{i} = fullfile(savepath, [savename, '_', stages{i}, '_new_', name, ext]);
    end
end
end
