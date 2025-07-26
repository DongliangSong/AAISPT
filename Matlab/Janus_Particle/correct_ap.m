% For huihui

azimuth = data(:,1);
polar = data(:,2);
dif = abs(diff(azimuth));

new = azimuth;
while max(dif) > 180
    for i = 1:length(dif)
        if dif(i) > 180
            if azimuth(i) < azimuth(i+1)
                azimuth(i+1) = azimuth(i+1) - 180;
                polar(i+1) = 180 - polar(i+1);
            else
                azimuth(i+1) = azimuth(i+1) + 180;
                polar(i+1) = 180 - polar(i+1);
            end
            dif = abs(diff(azimuth));
        end
    end
end

