function options = SPECTParameters(options)
%SPECTParameters Computes the necessary variables for projector types 2 and 6
%(SPECT)
%   Computes the PSF standard deviations for the current SPECT collimator
%   if omitted. Also computes the interaction planes with the rotated
%   projector and the projection angles.
%
% Copyright (C) 2022-2025 Ville-Veikko Wettenhovi, Matti Kortelainen, Niilo Saarlemo
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program. If not, see <https://www.gnu.org/licenses/>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ismember(options.projector_type, [1, 11, 12, 2, 21, 22]) % Collimator modelling, ray tracing projectors
    if numel(options.rayShiftsDetector) == 0
        options.rayShiftsDetector = [0; 0];
        options.rayShiftsDetector = repmat(options.rayShiftsDetector, [options.nRays, options.nRowsD, options.nColsD, options.nProjections]);

        if options.colFxy == 0 && options.colFz == 0 % Pinhole collimator
            dx = linspace(-(options.nRowsD/2-0.5)*options.dPitchX, (options.nRowsD/2-0.5)*options.dPitchX, options.nRowsD);
            dy = linspace(-(options.nColsD/2-0.5)*options.dPitchY, (options.nColsD/2-0.5)*options.dPitchY, options.nColsD);
            
            for ii = 1:options.nRowsD
                for jj = 1:options.nColsD
                    for kk = 1:options.nRays
                        options.rayShiftsDetector(2*(kk-1)+1, ii, jj, :) = -dx(ii);
                        options.rayShiftsDetector(2*(kk-1)+2, ii, jj, :) = -dy(jj);
                    end
                end
            end
        end
    end
    if numel(options.rayShiftsSource) == 0
        options.rayShiftsSource = [0; 0];
        options.rayShiftsSource = repmat(options.rayShiftsSource, [options.nRays, options.nRowsD, options.nColsD, options.nProjections]);

        if options.nRays > 1 % Multiray shifts
            nRays = sqrt(options.nRays);
            [tmp_x, tmp_y] = meshgrid(linspace(-0.5, 0.5, nRays));
            if options.colFxy == 0 && options.colFz == 0 % Pinhole collimator
                tmp_x = options.dPitchX * tmp_x;
                tmp_y = options.dPitchY * tmp_y;
            elseif ismember(options.colFxy, [-Inf, Inf]) && ismember(options.colFz, [-Inf, Inf]) % Parallel-hole collimator
                tmp_x = options.colR * tmp_x;
                tmp_y = options.colR * tmp_y;
            end

            tmp_shift = reshape([tmp_x(:), tmp_y(:)].', 1, [])';

            for kk = 1:options.nRays
                options.rayShiftsSource(2*(kk-1)+1) = tmp_shift(2*(kk-1)+1);
                options.rayShiftsSource(2*(kk-1)+2) = tmp_shift(2*(kk-1)+2);
            end
        end
    end
    if options.implementation == 2
        options.rayShiftsDetector = single(options.rayShiftsDetector(:));
        options.rayShiftsSource = single(options.rayShiftsSource(:));
    end
end
if ismember(options.projector_type, [12, 2, 21, 22]) % Orthogonal distance ray tracer
    % Frey, E. C., & Tsui, B. M. W. (n.d.). Collimator-Detector Response Compensation in SPECT. Quantitative Analysis in Nuclear Medicine Imaging, 141–166. doi:10.1007/0-387-25444-7_5 
    options.coneOfResponseStdCoeffA = 2*options.colR/options.colL; % See equation (6) of book chapter
    options.coneOfResponseStdCoeffB = 2*options.colR/options.colL*(options.colL+options.colD+options.cr_p/2);
    options.coneOfResponseStdCoeffC = options.iR;
    % Now the collimator response FWHM is sqrt((az+b)^2+c^2) where z is distance along detector element normal vector
end
if options.projector_type == 6
    DistanceToFirstRow = 0.5*options.dx;
    Distances = repmat(DistanceToFirstRow,1,options.Nx*4)+repmat((0:double(options.Nx*4)-1)*double(options.dx),length(DistanceToFirstRow),1);
    Distances = Distances-options.colL-options.colD; %these are distances to the actual detector surface

    if (~isfield(options,'gFilter'))
        if ~isfield(options, 'sigmaZ')
            Rg = 2*options.colR*(options.colL+options.colD+(Distances)+options.cr_p/2)/options.colL; %Anger, "Scintillation Camera with Multichannel Collimators", J Nucl Med 5:515-531 (1964)
            Rg(Rg<0) = 0;
            FWHMrot = 1;

            FWHM = sqrt(Rg.^2+options.iR^2);
            FWHM_pixel = FWHM/options.dx;
            expr = FWHM_pixel.^2-FWHMrot^2;
            expr(expr<=0) = 10^-16;
            FWHM_WithinPlane = sqrt(expr);

            %Parametrit CDR-mallinnukseen
            options.sigmaZ = FWHM_pixel./(2*sqrt(2*log(2)));
            options.sigmaXY = FWHM_WithinPlane./(2*sqrt(2*log(2)));
        end
        maxI = max([options.Nx(1), options.Ny(1), options.Nz(1)]);
        y = double(double(maxI) / 2 - 1:-1:-double(maxI) / 2 + 1);
        x = double(double(maxI) / 2 - 1:-1:-double(maxI) / 2 + 1);
        % xx = repmat(x', 1,options.Nz);
        xx = repmat(x', 1,size(x,2));
        yy = repmat(y, size(xx,1),1);
        if ~isfield(options, 'sigmaXY')
            s1 = double(repmat(permute(options.sigmaZ.^2,[4 3 2 1]), size(xx,1), size(yy,2), 1));
            options.gFilter = (1 / (2*pi*s1).*exp(-(xx.^2 + yy.^2)./(2*s1)));
        else
            s1 = double(repmat(permute(options.sigmaZ,[4 3 2 1]), size(xx,1), size(yy,2), 1));
            s2 = double(repmat(permute(options.sigmaXY,[4 3 2 1]), size(xx,1), size(yy,2), 1));
            options.gFilter = exp(-(xx.^2./(2*s1.^2) + yy.^2./(2*s2.^2)));
        end
        [rowE,colE] = find(options.gFilter(:,:,end/4) > 1e-6);
        [rowS,colS] = find(options.gFilter(:,:,end/4) > 1e-6);
        rowS = min(rowS);
        colS = min(colS);
        rowE = max(rowE);
        colE = max(colE);
        options.gFilter = options.gFilter(rowS:rowE,colS:colE,:,:);
        options.gFilter = options.gFilter ./ sum(sum(options.gFilter));
    end

    panelTilt = options.swivelAngles - options.angles + 180;
    options.blurPlanes = (options.FOVa_x/2 - (options.radiusPerProj .* cosd(panelTilt) - options.CORtoDetectorSurface)) / options.dx; % PSF shift
    options.blurPlanes2 = options.radiusPerProj .* sind(panelTilt) / options.dx; % Panel shift

    if options.implementation == 2
        options.blurPlanes = int32(options.blurPlanes);
        options.blurPlanes2 = int32(options.blurPlanes2);
    end
    if ~isfield(options,'angles') || numel(options.angles) == 0
        options.angles = (repelem(options.startAngle, options.nProjections / options.nHeads, 1) + repmat((0:options.angleIncrement:options.angleIncrement * (options.nProjections / options.nHeads - 1))', options.nHeads, 1) + options.offangle);
    end
    options.uu = 1;
    options.ub = 1;
    if options.useSingles
        options.gFilter = single(options.gFilter);
        options.angles = single(options.angles);
        options.swivelAngles = single(options.swivelAngles);
        options.radiusPerProj = single(options.radiusPerProj);
    end
end
end