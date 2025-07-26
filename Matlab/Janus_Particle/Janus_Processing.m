function Janus_Processing(path,savepath,plot_trace,plot_vector)

% Batch analysis of the motion trajectories of Janus particles, and save to the specified folder.

% path: The path to the original data.
% savepath: The path to the results.
% plot_trace: Whether to plot the trajectory? Type true if plotting, otherwise type [].
% plot_vector: Whether to plot the vector between two particles? Type true if plotting, otherwise type [].

%   example : 
%   path = 'D:\TrajSeg-Cls\Exp Data\YanYu';
%   savepath = 'D:\TrajSeg-Cls\Exp Data\YanYu\Results';
%   Janus_Processing(path,savepath,[],[]);

if isempty(plot_trace)
    plot_trace = false;
end

if isempty(plot_vector)
    plot_vector = false;
end


NP1_len = 200;    % large particle size.
NP2_len = 40;     % small particle size.
NP_len = (NP1_len + NP2_len)/2;    % radius
pixel_size = 106;     % pixel size


dirs = dir([path,'\*']);
dirs(1:2,:) = [];
dirs(end-2:end,:) = [];
num_dir = length(dirs);

count = 1;
for i = 1:num_dir
    files = dir(fullfile(dirs(i).folder,dirs(i).name));
    files(1:2,:) = [];
    num_file = length(files);
    
    try
        for j = 1:num_file
            filename = fullfile(files(j).folder,files(j).name);
            [folder,name,ext] = fileparts(filename);
            parts = strsplit(folder,'\');
    
            if strcmp(ext,'.opj') || strcmp(ext,'.opju')
                continue
            end
    
            % load data
            data = table2array(readtable(filename));
            num = size(data,1);
            time = data(:,5);

            if unique(isnan(data(:,3)))
                loc = data(:,1:2);

                % Save results
                ind = linspace(1,num,num);
                Table = array2table([ind',loc], 'VariableNames', {'Index','X', 'Y'});
                writetable(Table, fullfile(savepath,[num2str(count,'%03d'),'-',name,'_Single_track.xlsx']));
                count = count + 1;
                continue
            end

            loc = data(:,1:4);
            x1 = loc(:,1) * 1000;   % Convert to nm unit.
            y1 = loc(:,2) * 1000;
            x2 = loc(:,3) * 1000;
            y2 = loc(:,4) * 1000;
    
            % Calculate the in-plane angle (phi) between the projection of the vector onto the xy-plane and the x-axis.
            delta_x = x2 - x1;
            delta_y = y2 - y1;
            phi = atan2d(delta_y,delta_x);

            % Calculate the out-plane angle (theta) between the vector and the z-axis.
            projection_length = sqrt(delta_x.^2 + delta_y.^2);
            theta = real(asind(projection_length ./ NP_len));   % If the projection length is greater than 120nm, it is set to 120nm, which is equivalent to taking the real part of the angle.
            
            % Save results
            ind = linspace(1,num,num);
            Table = array2table([ind',x1,y1,x2,y2,phi,theta], 'VariableNames', {'Index','X1','Y1','X2','Y2','Azimuth', 'Polar'});
            
            writetable(Table, fullfile(savepath,[num2str(count,'%03d'),'-',name,'_results.xlsx']));
            count = count + 1;

            if plot_trace==true
                % Plot trace
                figure
                plot(loc(:,1),loc(:,2),'r','LineWidth',2)   % large particle
                hold on
                plot(loc(:,3),loc(:,4),'g','LineWidth',2)   % small particle
                hold off
                axis equal
                xlabel('X (nm)')
                ylabel('Y (nm)')
                box off
                set(gca,'LineWidth', 2,'FontSize', 14, 'FontWeight', 'bold')
                saveas(gcf, fullfile(savepath,[parts{4},'-',name,'_figure.png']));
            end
                
            if plot_vector==true
                % Draw vectors between particles.
                figure
                for k = 1:size(loc, 1)
                    x = [loc(k, 1), loc(k, 3)];
                    y = [loc(k, 2), loc(k, 4)];
                    plot(x, y, 'b','LineWidth',2);
                    hold on;
                
                    arrow_pos_x = loc(k, 3);
                    arrow_pos_y = loc(k, 4);
                    quiver(loc(k, 1), loc(k, 2), ...
                           arrow_pos_x - loc(k, 1), arrow_pos_y - loc(k, 2), ...
                           0, 'MaxHeadSize', 1);
                end
                
                axis equal;
                xlabel('X (nm)');
                ylabel('Y (nm)');
                title('Vector direction');
                grid on;
                set(gca,'LineWidth', 2,'FontSize', 14, 'FontWeight', 'bold')
                saveas(gcf, fullfile(savepath,'plot_figure.png'));
            end
        end
    catch e
        disp(e.message);
        disp(filename);
    end
end
end

