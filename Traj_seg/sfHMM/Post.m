

ranges = [1, 1248; 1248, 1833; 1833, 2050;2050,2512; 2512,2727; 2727, 3216; 3216, 3502];

figure
hold on

colors = ['r', 'g', 'b','y','c','m','k']; % Define colors to use
for i = 1:size(ranges, 1)
    range = ranges(i, 1):ranges(i, 2);
    plot3(xyz(range, 1), xyz(range, 2), xyz(range, 3), colors(mod(i-1, length(colors))+1),'LineWidth',2);
    pause(0.5)
end

hold off
set(gca,'LineWidth',2)

axis on

figure
hold on
for i = 1:size(ranges, 1)
    range = ranges(i, 1):ranges(i, 2);
    plot3(xyz(range, 1), xyz(range, 2), xyz(range, 3),'k','LineWidth',2);
    pause(0.5)
end

hold off
set(gca,'LineWidth',2)

axis on
