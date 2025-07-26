function [correctedAzimuth, correctedPolar] = correctAngles(azimuth, polar)
% CORRECTANGLES Corrects azimuth and polar angles to ensure the difference 
% between adjacent azimuth values does not exceed 180 degrees
%   [correctedAzimuth, correctedPolar] = correctAngles(azimuth, polar, logEnabled)
%   - azimuth: Input azimuth angle vector
%   - polar: Input polar angle vector
%   - correctedAzimuth: Corrected azimuth angle
%   - correctedPolar: Corrected polar angle


% Initialize output
correctedAzimuth = azimuth;
correctedPolar = polar;
dif = abs(diff(correctedAzimuth));

% Loop to adjust angles until the maximum difference is less than or equal to 180 degrees
while max(dif) > 180
    for i = 1:length(dif)
        if dif(i) > 180
            if correctedAzimuth(i) < correctedAzimuth(i + 1)
                correctedAzimuth(i + 1) = correctedAzimuth(i + 1) - 180;
                correctedPolar(i + 1) = 180 - correctedPolar(i + 1);
            else
                correctedAzimuth(i + 1) = correctedAzimuth(i + 1) + 180;
                correctedPolar(i + 1) = 180 - correctedPolar(i + 1);
            end
            dif = abs(diff(correctedAzimuth));
        end
    end
end
end