function angle = Correct_angle(azimuth)
% Correct the azimuth angle

angle = azimuth;
dif = abs(diff(azimuth));

while max(dif) > 180
    for k = 1:length(dif)
        if dif(k) > 180
            if azimuth(k) < azimuth(k + 1)
                angle(k + 1) = angle(k + 1) - 360;
            else
                angle(k + 1) = angle(k + 1) + 360;
            end
            dif = abs(diff(angle));
        end
    end
end
end