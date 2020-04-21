function [Sin, gaps] = gapFilling(options, Sin, xp, yp, llo, gaps)
%GAPFILLING Performs gap filling to pseudo detector sinograms
%   Dedicated function for sinogram gap filling. If pseudo detectors are
%   used (e.g. Biograph mCT), these detectors don't actually receive any
%   measurements. This causes zero measurements when no mashing and reduced
%   number of measurements with mashing. The gap filling interpolation
%   attempts to fix these gaps.
% 
%   Two types of interpolation are available. The built-in fillmissing or
%   the function inpaint_nans from File exchange: 
%   https://se.mathworks.com/matlabcentral/fileexchange/4551-inpaint_nans
%
% INPUTS:
%   options = Machine and sinogram properties
%   Sin = The original sinogram
%   xp, yp = Sinogram coordinates
%   llo = Current timestep
%   gaps = Location of the gaps to be filled (used for subsequent
%   timesteps)
%
% OUTPUTS:
%   Sin = The gap-filled sinogram
%   gaps = The location of the sinogram gaps for subsequent timesteps
%
% See also fillmissing, inpaint_nans, form_sinograms


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2020 Ville-Veikko Wettenhovi, Samuli Summala
%
% This program is free software: you can redistribute it and/or modify it
% under the terms of the GNU General Public License as published by the
% Free Software Foundation, either version 3 of the License, or (at your
% option) any later version. 
%
% This program is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
% Public License for more details. 
%
% You should have received a copy of the GNU General Public License along
% with this program. If not, see <https://www.gnu.org/licenses/>. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mashing = options.det_w_pseudo / options.Nang / 2;
if mashing == 1
    apu = sum(Sin,3);
    gaps = apu == 0;
else
    if llo == 1
        options.Nang = options.Nang * mashing;
        pseudot = options.pseudot;
        ringsp = options.rings;
        temp = pseudot;
        if ~isempty(temp) && temp > 0
            for kk = uint32(1) : temp
                pseudot(kk) = uint32(options.cryst_per_block + 1) * kk;
            end
        elseif temp == 0
            pseudot = [];
        end
        [~, ~, i, j, accepted_lors] = sinogram_coordinates_2D(options, xp, yp);
        
        L = zeros(sum(1:options.det_w_pseudo),2,'int32');
        jh = int32(1);
        for kk = int32(1) : (options.det_w_pseudo)
            if exist('OCTAVE_VERSION','builtin') == 0 && exist('repelem', 'builtin') == 0
                L(jh:(jh + (options.det_w_pseudo) - kk),:) = [repeat_elem((kk), options.det_w_pseudo-(kk-1)), ((kk):options.det_w_pseudo)'];
            elseif exist('OCTAVE_VERSION','builtin') == 5
                L(jh:(jh + (options.det_w_pseudo) - kk),:) = [repelem((kk), options.det_w_pseudo-(kk-1)), ((kk):options.det_w_pseudo)'];
            else
                L(jh:(jh + (options.det_w_pseudo) - kk),:) = [repelem((kk), options.det_w_pseudo-(kk-1))', ((kk):options.det_w_pseudo)'];
            end
            jh = jh + (options.det_w_pseudo) -kk + 1;
        end
        L(L(:,1) == 0,:) = [];
        
        L = L - 1;
        L = L(accepted_lors,:);
        
        L = L + 1;
        
        locci = true(size(L,1),1);
        for dd = 1 : size(L,1)
            if mod(L(dd,1),14) == 0
                locci(dd) = false;
            end
        end
        ykkoset = zeros(size(L,1),1);
        ykkoset(locci) = 1;
        ff = accumarray([i j], ykkoset, [options.Ndist options.Nang]);
        locci = true(size(L,1),1);
        for dd = 1 : size(L,1)
            if mod(L(dd,2),14) == 0
                locci(dd) = false;
            end
        end
        ykkoset = zeros(size(L,1),1);
        ykkoset(locci) = 1;
        ff2 = accumarray([i j], ykkoset, [options.Ndist options.Nang]);
        uusf = ff + ff2;
        uusf(uusf == 1) = 0;
        uusf(uusf > 0) = 1;
        apu = uusf;
        apu(apu == 0) = NaN;
        apu(~isnan(apu)) = 0;
        apu(isnan(apu)) = 1;
        for kk = 1 : mashing - 1
            apu = apu(:,1:mashing:end) + apu(:,2:mashing:end);
        end
        gaps = apu > 0;
        apu = gaps(end,:);
        gaps(2:end,:) = gaps(1:end-1,:);
        gaps(1,:) = apu;
    end
end
if strcmp('fillmissing',options.gap_filling_method)
    Sin = single(Sin);
    for kk = 1 : size(Sin,3)
        apu = Sin(:,:,kk);
        appa = false(size(apu));
%         appa(apu == 0) = true;
        gappa = logical(gaps + appa);
        apu(gappa) = NaN;
        jelppi1 = fillmissing(apu, options.interpolation_method_fillmissing);
        jelppi2 = fillmissing(apu', options.interpolation_method_fillmissing)';
        Sin(:,:,kk) = (jelppi1 + jelppi2) / 2;
    end
%     Sin(Sin < 0) = 0;
    th = floor(options.Ndist * 0.05);
    Sin(1:th - 1,:,:) = repmat(Sin(th,:,:),th - 1, 1, 1);
    Sin(end - th + 2 : end,:,:) = repmat(Sin(end - th + 1,:,:), th - 1, 1, 1);
elseif strcmp('inpaint_nans',options.gap_filling_method)
    Sin = single(Sin);
    for kk = 1 : size(Sin,3)
        apu = Sin(:,:,kk);
        appa = false(size(apu));
%         appa(apu == 0) = true;
        gappa = logical(gaps + appa);
        apu(gappa) = NaN;
%         apu(gaps) = NaN;
        Sin(:,:,kk) = single(inpaint_nans(double(apu), options.interpolation_method_inpaint));
    end
%     Sin(Sin < 0) = 0;
    th = floor(options.Ndist * 0.05);
    Sin(1:th - 1,:,:) = repmat(Sin(th,:,:),th - 1, 1, 1);
    Sin(end - th + 2 : end,:,:) = repmat(Sin(end - th + 1,:,:), th - 1, 1, 1);
else
    warning('Unsupported gap filling method! No gap filling was performed!')
end
if options.verbose
    disp('Gap filling completed')
end
end

