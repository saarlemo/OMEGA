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
DistanceToFirstRow = options.radiusPerProj-(double(options.Nx)/2-0.5)*double(options.dx);
Distances = repmat(DistanceToFirstRow,1,options.Nx)+repmat((0:double(options.Nx)-1)*double(options.dx),length(DistanceToFirstRow),1);
Distances = Distances-options.colL-options.colD; %these are distances to the actual detector surface

if (~isfield(options,'gFilter') || projector_type == 2) % gFilter not supported by projector_type == 2
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
    [~,ind] = max(options.radiusPerProj);
    [rowE,colE] = find(options.gFilter(:,:,end,ind) > 1e-6);
    [rowS,colS] = find(options.gFilter(:,:,end,ind) > 1e-6);
    rowS = min(rowS);
    colS = min(colS);
    rowE = max(rowE);
    colE = max(colE);
    % rowS = find(options.gFilter(round(size(options.gFilter,1)/2),:,end,ind) > 1e-6,1,'first');
    % colE = find(options.gFilter(:,round(size(options.gFilter,2)/2),end,ind) > 1e-6,1,'last');
    % colS = find(options.gFilter(:,round(size(options.gFilter,2)/2),end,ind) > 1e-6,1,'first');
    options.gFilter = options.gFilter(rowS:rowE,colS:colE,:,:);
    options.gFilter = options.gFilter ./ sum(sum(options.gFilter));

    if options.projector_type == 2 % Calculate linear coefficients for standard deviation, it is assumed that sigma=st where t is the distance from detector to the orthogonal projection of current voxel and s is a constant determined below.
        distIdx = Distances(1, :) >= 100; % The relation between standard deviation and distance is approximately linear in this range
        tmpDist = Distances(1, distIdx); % Distances above the threshold
        tmpStd = options.sigmaXY(1, distIdx); % Standard deviations above the threshold
        options.coneOfResponseStdCoeff = (tmpDist * tmpDist') \ (tmpDist * tmpStd'); % Least squares line fit
        %disp(stdCoeff)
    end
end

if options.projector_type == 6 % Not required for projector_type == 2
    [~, options.blurPlanes] = max(Distances>0,[],2);
    if options.implementation == 2
        options.blurPlanes = uint32(options.blurPlanes - 1);
    end
    if ~isfield(options,'angles') || numel(options.angles) == 0
        options.angles = (repelem(options.startAngle, options.nProjections / options.nHeads, 1) + repmat((0:options.angleIncrement:options.angleIncrement * (options.nProjections / options.nHeads - 1))', options.nHeads, 1) + options.offangle);
    end
    options.uu = 1;
    options.ub = 1;
    if max(abs(options.angles(:))) > 10 * pi && options.implementation == 2
        options.angles = options.angles / 180 * pi;
    end
    if options.flip_image
        options.angles = -(options.angles);
    end
end
if options.implementation == 2 || options.useSingles
    options.gFilter = single(options.gFilter);
    options.angles = single(options.angles);
end
end