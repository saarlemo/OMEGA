function [normalization_correction, randoms_correction, options] = loadCorrections(options, RandProp, ScatterProp)
%% SET UP CORRECTIONS
% This function sets up (loads) the necessary corrections that are applied.
% Included are attenuation correction, normalization correction and randoms
% and/or scatter correction. The variables used for correction are saved in
% the options-struct that also contains the input variables except for the
% number of rings in the current machine.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2022-2024 Ville-Veikko Wettenhovi
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

folder = fileparts(which('set_up_corrections.m'));
folder = [folder(1:end-(6 + 8)), 'mat-files/'];
folder = strrep(folder, '\','/');

if ~isfield(options,'reconstruct_trues')
    options.reconstruct_trues = false;
end
if ~isfield(options,'reconstruct_scatter')
    options.reconstruct_scatter = false;
end
if ~isfield(options,'TOF_bins') || options.TOF_bins == 0
    options.TOF_bins = 1;
end
if ~isfield(options,'use_machine')
    options.use_machine = 0;
end

if ~options.use_raw_data && options.randoms_correction
    if options.partitions == 1
        load_string = [options.machine_name '_' options.name '_sinograms_combined_static_' num2str(options.Ndist) 'x' num2str(options.Nang) 'x' ...
            num2str(options.TotSinos) '_span' num2str(options.span)];
        if options.use_machine == 0
            sinoFile = [load_string '.mat'];
        elseif options.use_machine == 1
            sinoFile = [load_string '_listmode.mat'];
        elseif options.use_machine == 2
            sinoFile = [load_string '_machine_sinogram.mat'];
        elseif options.use_machine == 3
            sinoFile = [load_string '_listmode_sinogram.mat'];
        end
    else
        load_string = [options.machine_name '_' options.name '_sinograms_combined_' num2str(options.partitions) 'timepoints_for_total_of_' ...
            num2str(options.tot_time) 's_' num2str(options.Ndist) 'x' num2str(options.Nang) 'x' num2str(options.TotSinos) '_span' ...
            num2str(options.span)];
        load_string2 = [options.machine_name '_' options.name '_sinograms_combined_' num2str(options.partitions) 'timepoints_for_total_of_ ' ...
            num2str(options.tot_time) 's_' num2str(options.Ndist) 'x' num2str(options.Nang) 'x' num2str(options.TotSinos) '_span' ...
            num2str(options.span)];
        if options.use_machine == 0
            sinoFile = [load_string '.mat'];
            if exist(sinoFile, 'file') == 0
                sinoFile = [load_string2 '.mat'];
            end
        elseif options.use_machine == 1
            sinoFile = [load_string '_listmode.mat'];
            if exist(sinoFile, 'file') == 0
                sinoFile = [load_string2 '_listmode.mat'];
            end
        elseif options.use_machine == 2
            sinoFile = [load_string '_machine_sinogram.mat'];
            if exist(sinoFile, 'file') == 0
                sinoFile = [load_string2 '_machine_sinogram.mat'];
            end
        elseif options.use_machine == 3
            sinoFile = [load_string '_listmode_sinogram.mat'];
            if exist(sinoFile, 'file') == 0
                sinoFile = [load_string2 '_listmode_sinogram.mat'];
            end
        end
    end
end

if options.attenuation_correction
    if ~isfield(options,'vaimennus') || isempty(options.vaimennus)
        if ~isempty(options.attenuation_datafile) && strcmp(options.attenuation_datafile(end-2:end), 'mhd')
            [options.vaimennus, apuStruct] = loadMetaImage(options.attenuation_datafile);
            options.vaimennus = options.vaimennus(:);
            if options.CT_attenuation
                if round(apuStruct.EleSpacing(1)*100)/100 > round(options.FOVa_x(1) / double(options.Nx(1))*100)/100 ||  round(apuStruct.EleSpacing(1)*100)/100 < round(options.FOVa_x(1) / double(options.Nx(1))*100)/100
                    options.vaimennus = options.vaimennus * (apuStruct.EleSpacing(1) / (options.FOVa_x(1) / double(options.Nx(1))));
                end
            end
        elseif ~isempty(options.attenuation_datafile)
            data = load(options.attenuation_datafile);
            variables = fieldnames(data);
            options.vaimennus = double(data.(variables{1}));
            if options.CT_attenuation
                options.vaimennus = options.vaimennus(:) / 10;
            end
            clear data
        else
            [options.file, options.fpath] = uigetfile('*.*','Select attenuation file');
            if isequal(options.file, 0)
                error('No file was selected')
            end
            nimi = [options.fpath options.file];
            if strcmp(nimi(end-2:end), 'mhd')
                [options.vaimennus, ~] = loadMetaImage(nimi);
            elseif strcmp(nimi(end-2:end), 'h33')
                [options.vaimennus] = loadInterfile(nimi);
            else
                data = load(nimi);
                variables = fieldnames(data);
                options.vaimennus = double(data.(variables{1}));
                if options.CT_attenuation
                    options.vaimennus = options.vaimennus(:) / 10;
                end
                clear data
            end
        end
    end
    if options.CT_attenuation
        if size(options.vaimennus,1) ~= options.Nx(1) || size(options.vaimennus,2) ~= options.Ny(1) || size(options.vaimennus,3) ~= options.Nz(1)
            if size(options.vaimennus,1) ~= options.N(1)
                error('Error: Attenuation data is of different size than the reconstructed image')
            end
        end
        if isfield(options,'rotateAttImage') && options.rotateAttImage ~= 0
            atn = reshape(options.vaimennus, options.Nx(1), options.Ny(1), options.Nz(1));
            atn = rot90(atn,options.rotateAttImage);
            options.vaimennus = atn(:);
        end
        if isfield(options,'flipAttImageXY') && options.flipAttImageXY
            atn = reshape(options.vaimennus, options.Nx(1), options.Ny(1), options.Nz(1));
            atn = fliplr(atn);
            options.vaimennus = atn(:);
        end
        if isfield(options,'flipAttImageZ') && options.flipAttImageZ
            atn = reshape(options.vaimennus, options.Nx(1), options.Ny(1), options.Nz(1));
            atn = flip(atn,3);
            options.vaimennus = atn(:);
        end
    end
    if options.implementation == 2 || options.implementation == 3 || options.implementation == 5 || options.useSingles
        options.vaimennus = single(options.vaimennus);
    else
        options.vaimennus = double(options.vaimennus);
    end
else
    if options.implementation == 2 || options.implementation == 3 || options.implementation == 5 || options.useSingles
        options.vaimennus = single(0);
    else
        options.vaimennus = 0;
    end
end

if options.scatter_correction && options.normalize_scatter
    if ~isfield(options,'normalization')
        if options.use_user_normalization
            [file, fpath] = uigetfile({'*.nrm;*.mat'},'Select normalization datafile');
            if isequal(file, 0)
                error('No file was selected')
            end
            if any(strfind(file, '.nrm'))
                fid = fopen(fullfile(fpath, file));
                options.normalization = fread(fid, inf, 'single=>single',0,'l');
                fclose(fid);
                options.InveonNorm = true;
            else
                data = load(fullfile(fpath, file));
                variables = fieldnames(data);
                options.normalization = data.(variables{1});
                clear data
                if numel(options.normalization) ~= numel(options.SinM{1})
                    error('Size mismatch between the current data and the normalization data file')
                end
            end
            options.normalization = options.normalization(:);
        else
            if options.use_raw_data
                norm_file = [folder options.machine_name '_normalization_listmode.mat'];
            else
                norm_file = [folder options.machine_name '_normalization_' num2str(options.Ndist) 'x' num2str(options.Nang) '_span' num2str(options.span) '.mat'];
            end
            if exist(norm_file, 'file') == 2
                options.normalization = loadStructFromFile(norm_file,'normalization');
            else
                error('Normalization correction selected, but no normalization data found')
            end
            options.normalization = options.normalization(:);
        end
        if ~options.use_raw_data && options.NSinos ~= options.TotSinos
            options.normalization = options.normalization(1 : options.Ndist * options.Nang *options.NSinos);
        end
        if options.sampling > 1 && options.corrections_during_reconstruction
            options.normalization = reshape(options.normalization, options.Ndist, options.Nang, options.NSinos);
            options.normalization = interpolateSinog(options.normalization, options.sampling, options.Ndist, options.partitions, options.sampling_interpolation_method);
            options.normalization = options.normalization(:);
        end
    end
end

% Apply randoms and scatter (if available) correction
% Apply randoms smoothing/variance reduction
if (options.randoms_correction || options.scatter_correction) && options.corrections_during_reconstruction && ~options.reconstruct_trues...
        && ~options.reconstruct_scatter
    randoms_correction = true;
    r_exist = isfield(options,'SinDelayed') && numel(options.SinDelayed) > 1;
    s_exist = isfield(options,'ScatterC') && numel(options.ScatterC) > 1;
    if r_exist && isfield(options,'SinDelayed') && ~iscell(options.SinDelayed) && numel(options.SinDelayed) == 1 && randoms_correction
        r_exist = false;
    elseif r_exist && isfield(options,'SinDelayed') && iscell(options.SinDelayed) && numel(options.SinDelayed{1}) == 1 && randoms_correction
        r_exist = false;
    end
    if exist('RandProp','var') == 0 || isempty(RandProp)
        RandProp.smoothing = false;
        RandProp.variance_reduction = false;
    end
    if exist('ScatterProp','var') == 0 || isempty(ScatterProp)
        ScatterProp.smoothing = false;
        ScatterProp.variance_reduction = false;
        ScatterProp.normalization = false;
    end
    if r_exist && randoms_correction
        if (~options.variance_reduction && RandProp.variance_reduction) || (~options.randoms_smoothing && RandProp.smoothing)
            options.SinDelayed = loadStructFromFile(sinoFile, 'raw_SinDelayed');
            RandProp.variance_reduction = false;
            RandProp.smoothing = false;
        end
    end
    if options.randoms_correction && ~r_exist
        options = loadDelayedData(options);
        r_exist = true;
    end
    if options.scatter_correction && ((~isfield(options,'ScatterC') || numel(options.ScatterC) == 0) || (ScatterProp.normalization && ~options.normalize_scatter) || ...
            (ScatterProp.smoothing && ~options.scatter_smoothing) || (ScatterProp.variance_reduction && ~options.scatter_variance_reduction))
        if ScatterProp.normalization && ~options.normalize_scatter
            warning('Scatter data is normalized, but no normalized scatter selected. Input non-normalized scatter data or put options.normalize_scatter = true!')
        end
        if ScatterProp.smoothing && ~options.scatter_smoothing
            warning('Scatter data is smoothed, but no scatter smoothing selected. Input non-smoothed scatter data or put options.scatter_smoothing = true!')
        end
        if ScatterProp.variance_reduction && ~options.scatter_variance_reduction
            warning('Scatter data is variance corrected, but no scatter variance correction selected. Input non-variance corrected scatter data or put options.scatter_variance_reduction = true!')
        end
        options = loadScatterData(options);
        s_exist = true;
    end
    if r_exist || s_exist
        if r_exist && iscell(options.SinDelayed)
            for kk = 1 : options.partitions
                if r_exist
                    if options.variance_reduction && ~RandProp.variance_reduction
                        options.SinDelayed{kk} = Randoms_variance_reduction(single(options.SinDelayed{kk}), options);
                    end
                    if options.randoms_smoothing && ~RandProp.smoothing
                        options.SinDelayed{kk} = randoms_smoothing(single(options.SinDelayed{kk}), options);
                    end
                    options.SinDelayed{kk} = options.SinDelayed{kk}(:);
                    if options.use_raw_data == false && options.NSinos ~= options.TotSinos
                        options.SinDelayed{kk} = options.SinDelayed{kk}(1:options.NSinos*options.Ndist*options.Nang);
                    end
                end
                if options.scatter_correction
                    if sum(size(options.ScatterC)) > 1 && ~iscell(options.ScatterC)
                        if size(options.ScatterC,1) ~= options.Ndist && ~options.use_raw_data
                            options.ScatterC = permute(options.ScatterC,[2 1 3]);
                        end
                        if options.normalization_correction && options.normalize_scatter && ~ScatterProp.normalization
                            if isfield(options,'InveonScatter') && isfield(options,'InveonNorm')
                                options.ScatterC = single(options.ScatterC) .* reshape(options.normalization, size(options.ScatterC, 1), size(options.ScatterC, 2), size(options.ScatterC, 3));
                            else
                                options.ScatterC = single(options.ScatterC) .* reshape(1 ./ options.normalization, size(options.ScatterC, 1), size(options.ScatterC, 2), size(options.ScatterC, 3));
                            end
                        end
                        if options.scatter_variance_reduction && ~ScatterProp.variance_reduction
                            options.ScatterC = Randoms_variance_reduction(single(options.ScatterC), options);
                        end
                        if options.scatter_smoothing && ~ScatterProp.smoothing
                            options.ScatterC = randoms_smoothing(single(options.ScatterC), options);
                        end
                    elseif iscell(options.ScatterC)
                        if sum(size(options.ScatterC{1})) > 1
                            if size(options.ScatterC{kk},1) ~= options.Ndist && ~options.use_raw_data
                                options.ScatterC{kk} = permute(options.ScatterC{kk},[2 1 3]);
                            end
                        end
                        if options.normalization_correction && options.normalize_scatter
                            if isfield(options,'InveonScatter') && isfield(options,'InveonNorm')
                                options.ScatterC{kk} = single(options.ScatterC{kk}) .* reshape(options.normalization, size(options.ScatterC{kk}, 1), size(options.ScatterC{kk}, 2), size(options.ScatterC{kk}, 3));
                            else
                                options.ScatterC{kk} = single(options.ScatterC{kk}) .* reshape(1 ./ options.normalization, size(options.ScatterC{kk}, 1), size(options.ScatterC{kk}, 2), size(options.ScatterC{kk}, 3));
                            end
                        end
                        if options.scatter_variance_reduction && ~ScatterProp.variance_reduction
                            options.ScatterC{kk} = Randoms_variance_reduction(single(options.ScatterC{kk}), options);
                        end
                        if options.scatter_smoothing && ~ScatterProp.smoothing
                            options.ScatterC{kk} = randoms_smoothing(single(options.ScatterC{kk}), options);
                        end
                    end
                    if r_exist
                        if options.implementation == 2 || options.implementation == 3 || options.implementation == 5 || options.implementation == 4
                            options.SinDelayed{kk} = single(options.SinDelayed{kk}) / options.TOF_bins + single(options.ScatterC{kk}(:));
                        else
                            options.SinDelayed{kk} = double(options.SinDelayed{kk}) / options.TOF_bins + double(options.ScatterC{kk}(:));
                        end
                    else
                        if options.implementation == 2 || options.implementation == 3 || options.implementation == 5 || options.implementation == 4
                            options.SinDelayed{kk} = single(options.ScatterC{kk}(:));
                        else
                            options.SinDelayed{kk} = double(options.ScatterC{kk}(:));
                        end
                    end
                elseif ~options.scatter_correction && options.TOF_bins > 1 && r_exist
                    if options.implementation == 2 || options.implementation == 3 || options.implementation == 5 || options.implementation == 4
                        options.SinDelayed{kk} = single(options.SinDelayed{kk} / options.TOF_bins);
                    else
                        options.SinDelayed{kk} = double(options.SinDelayed{kk} / options.TOF_bins);
                    end
                end
            end
        else
            if r_exist
                if options.variance_reduction && ~RandProp.variance_reduction
                    options.SinDelayed = Randoms_variance_reduction(single(options.SinDelayed), options);
                end
                if options.randoms_smoothing && ~RandProp.smoothing
                    options.SinDelayed = randoms_smoothing(single(options.SinDelayed), options);
                end
                options.SinDelayed = options.SinDelayed(:);
                if options.use_raw_data == false && options.NSinos ~= options.TotSinos
                    options.SinDelayed = options.SinDelayed(1:options.NSinos*options.Ndist*options.Nang);
                end
            end
            if options.scatter_correction && isfield(options,'ScatterC')
                if isempty(ScatterProp)
                    ScatterProp.smoothing = false;
                    ScatterProp.variance_reduction = false;
                    ScatterProp.normalization = false;
                end
                if sum(size(options.ScatterC)) > 1 && ~iscell(options.ScatterC)
                    if size(options.ScatterC,1) ~= options.Ndist && ~options.use_raw_data
                        options.ScatterC = permute(options.ScatterC,[2 1 3]);
                    end
                    if options.normalization_correction && options.normalize_scatter && ~ScatterProp.normalization
                        if isfield(options,'InveonScatter') && isfield(options,'InveonNorm')
                            options.ScatterC = single(options.ScatterC) .* reshape(options.normalization, size(options.ScatterC, 1), size(options.ScatterC, 2), size(options.ScatterC, 3));
                        else
                            options.ScatterC = single(options.ScatterC) .* reshape(1 ./ options.normalization, size(options.ScatterC, 1), size(options.ScatterC, 2), size(options.ScatterC, 3));
                        end
                    end
                    if options.scatter_variance_reduction && ~ScatterProp.variance_reduction
                        options.ScatterC = Randoms_variance_reduction(single(options.ScatterC), options);
                    end
                    if options.scatter_smoothing && ~ScatterProp.smoothing
                        options.ScatterC = randoms_smoothing(single(options.ScatterC), options);
                    end
                    if options.use_raw_data == false && options.NSinos ~= options.TotSinos
                        options.ScatterC = options.ScatterC(1:options.NSinos*options.Ndist*options.Nang);
                    end
                    if r_exist
                        if options.implementation == 2 || options.implementation == 3 || options.implementation == 5 || options.useSingles
                            options.SinDelayed = single(options.SinDelayed / options.TOF_bins) + single(options.ScatterC(:));
                        else
                            options.SinDelayed = double(options.SinDelayed / options.TOF_bins) + double(options.ScatterC(:));
                        end
                    else
                        if options.implementation == 2 || options.implementation == 3 || options.implementation == 5 || options.useSingles
                            options.SinDelayed = single(options.ScatterC(:));
                        else
                            options.SinDelayed = double(options.ScatterC(:));
                        end
                    end
                elseif iscell(options.ScatterC)
                    for kk = 1 : options.partitions
                        if sum(size(options.ScatterC{kk})) > 1
                            if size(options.ScatterC{kk},1) ~= options.Ndist && ~options.use_raw_data
                                options.ScatterC{kk} = permute(options.ScatterC{kk},[2 1 3]);
                            end
                        end
                        if options.normalization_correction && options.normalize_scatter && ~ScatterProp.normalization
                            if isfield(options,'InveonScatter') && isfield(options,'InveonNorm')
                                options.ScatterC{kk} = single(options.ScatterC{kk}) .* reshape(options.normalization, size(options.ScatterC{kk}, 1), size(options.ScatterC{kk}, 2), size(options.ScatterC{kk}, 3));
                            else
                                options.ScatterC{kk} = single(options.ScatterC{kk}) .* reshape(1 ./ options.normalization, size(options.ScatterC{kk}, 1), size(options.ScatterC{kk}, 2), size(options.ScatterC{kk}, 3));
                            end
                        end
                        if options.scatter_variance_reduction && ~ScatterProp.variance_reduction
                            options.ScatterC{kk} = Randoms_variance_reduction(single(options.ScatterC{kk}), options);
                        end
                        if options.scatter_smoothing && ~ScatterProp.smoothing
                            options.ScatterC{kk} = randoms_smoothing(single(options.ScatterC{kk}), options);
                        end
                        if options.use_raw_data == false && options.NSinos ~= options.TotSinos
                            options.ScatterC{kk} = options.ScatterC{kk}(1:options.NSinos*options.Ndist*options.Nang);
                        end
                        if r_exist
                            if options.implementation == 2 || options.implementation == 3 || options.implementation == 5 || options.useSingles
                                options.SinDelayed = single(options.SinDelayed / options.TOF_bins) + single(options.ScatterC{kk}(:));
                            else
                                options.SinDelayed = double(options.SinDelayed / options.TOF_bins) + double(options.ScatterC{kk}(:));
                            end
                        else
                            options.SinDelayed = cell(numel(options.ScatterC),1);
                            if options.implementation == 2 || options.implementation == 3 || options.implementation == 5 || options.useSingles
                                options.SinDelayed{kk} = single(options.ScatterC{kk}(:));
                            else
                                options.SinDelayed{kk} = double(options.ScatterC{kk}(:));
                            end
                        end
                    end
                end
            elseif ~options.scatter_correction && options.TOF_bins > 1 && r_exist
                if options.implementation == 2 || options.implementation == 3 || options.implementation == 5 || options.useSingles
                    options.SinDelayed = single(options.SinDelayed / options.TOF_bins);
                else
                    options.SinDelayed = double(options.SinDelayed / options.TOF_bins);
                end
            end
        end
    else
        if options.randoms_correction && iscell(options.SinDelayed)
            for kk = 1 : options.partitions
                if options.variance_reduction && ~RandProp.variance_reduction
                    options.SinDelayed{kk} = Randoms_variance_reduction(single(options.SinDelayed{kk}), options);
                end
                if options.randoms_smoothing && ~RandProp.smoothing
                    options.SinDelayed{kk} = randoms_smoothing(single(options.SinDelayed{kk}), options);
                end
                options.SinDelayed{kk} = options.SinDelayed{kk}(:);
                if options.use_raw_data == false && options.NSinos ~= options.TotSinos
                    options.SinDelayed{kk} = options.SinDelayed{kk}(1:options.NSinos*options.Ndist*options.Nang);
                end
                if options.scatter_correction
                    if sum(size(options.ScatterC)) > 1 && ~iscell(options.ScatterC)
                        if size(options.ScatterC,1) ~= options.Ndist && ~options.use_raw_data
                            options.ScatterC = permute(options.ScatterC,[2 1 3]);
                        end
                        if options.normalization_correction && options.normalize_scatter && ~ScatterProp.normalization
                            if isfield(options,'InveonScatter') && isfield(options,'InveonNorm')
                                options.ScatterC = single(options.ScatterC) .* reshape(options.normalization, size(options.ScatterC, 1), size(options.ScatterC, 2), size(options.ScatterC, 3));
                            else
                                options.ScatterC = single(options.ScatterC) .* reshape(1 ./ options.normalization, size(options.ScatterC, 1), size(options.ScatterC, 2), size(options.ScatterC, 3));
                            end
                        end
                        if options.scatter_variance_reduction && ~ScatterProp.variance_reduction
                            options.ScatterC = Randoms_variance_reduction(single(options.ScatterC), options);
                        end
                        if options.scatter_smoothing && ~ScatterProp.smoothing
                            options.ScatterC = randoms_smoothing(single(options.ScatterC), options);
                        end
                    elseif iscell(options.ScatterC)
                        if sum(size(options.ScatterC{1})) > 1
                            if size(options.ScatterC{kk},1) ~= options.Ndist && ~options.use_raw_data
                                options.ScatterC{kk} = permute(options.ScatterC{kk},[2 1 3]);
                            end
                        end
                        if options.normalization_correction && options.normalize_scatter && ~ScatterProp.normalization
                            if isfield(options,'InveonScatter') && isfield(options,'InveonNorm')
                                options.ScatterC{kk} = single(options.ScatterC{kk}) .* reshape(options.normalization, size(options.ScatterC{kk}, 1), size(options.ScatterC{kk}, 2), size(options.ScatterC{kk}, 3));
                            else
                                options.ScatterC{kk} = single(options.ScatterC{kk}) .* reshape(1 ./ options.normalization, size(options.ScatterC{kk}, 1), size(options.ScatterC{kk}, 2), size(options.ScatterC{kk}, 3));
                            end
                        end
                        if options.scatter_variance_reduction && ~ScatterProp.variance_reduction
                            options.ScatterC{kk} = Randoms_variance_reduction(single(options.ScatterC{kk}), options);
                        end
                        if options.scatter_smoothing && ~ScatterProp.smoothing
                            options.ScatterC{kk} = randoms_smoothing(single(options.ScatterC{kk}), options);
                        end
                    end
                    if options.implementation == 2 || options.implementation == 3 || options.implementation == 5 || options.useSingles
                        options.SinDelayed{kk} = single(options.SinDelayed{kk} / options.TOF_bins) + single(options.ScatterC{kk}(:));
                    else
                        options.SinDelayed{kk} = double(options.SinDelayed{kk} / options.TOF_bins) + double(options.ScatterC{kk}(:));
                    end
                elseif ~options.scatter_correction && options.TOF_bins > 1
                    if options.implementation == 2 || options.implementation == 3 || options.implementation == 5 || options.useSingles
                        options.SinDelayed{kk} = single(options.SinDelayed{kk}) / single(options.TOF_bins);
                    else
                        options.SinDelayed{kk} = double(options.SinDelayed{kk}) / options.TOF_bins;
                    end
                end
            end
        elseif options.randoms_correction
            if options.variance_reduction && ~RandProp.variance_reduction
                options.SinDelayed = Randoms_variance_reduction(single(options.SinDelayed), options);
            end
            if options.randoms_smoothing && ~RandProp.smoothing
                options.SinDelayed = randoms_smoothing(single(options.SinDelayed), options);
            end
            options.SinDelayed = options.SinDelayed(:);
            if options.use_raw_data == false && options.NSinos ~= options.TotSinos
                options.SinDelayed = options.SinDelayed(1:options.NSinos*options.Ndist*options.Nang);
            end
            if options.scatter_correction && isfield(options,'ScatterC')
                if sum(size(options.ScatterC)) > 1 && ~iscell(options.ScatterC)
                    if size(options.ScatterC,1) ~= options.Ndist && ~options.use_raw_data
                        options.ScatterC = permute(options.ScatterC,[2 1 3]);
                    end
                    if options.normalization_correction && options.normalize_scatter && ~ScatterProp.normalization
                        if isfield(options,'InveonScatter') && isfield(options,'InveonNorm')
                            options.ScatterC = single(options.ScatterC) .* reshape(options.normalization, size(options.ScatterC, 1), size(options.ScatterC, 2), size(options.ScatterC, 3));
                        else
                            options.ScatterC = single(options.ScatterC) .* reshape(1 ./ options.normalization, size(options.ScatterC, 1), size(options.ScatterC, 2), size(options.ScatterC, 3));
                        end
                    end
                    if options.scatter_variance_reduction && ~ScatterProp.variance_reduction
                        options.ScatterC = Randoms_variance_reduction(single(options.ScatterC), options);
                    end
                    if options.scatter_smoothing && ~ScatterProp.smoothing
                        options.ScatterC = randoms_smoothing(single(options.ScatterC), options);
                    end
                    if options.use_raw_data == false && options.NSinos ~= options.TotSinos
                        options.ScatterC = options.ScatterC(1:options.NSinos*options.Ndist*options.Nang);
                    end
                    if options.implementation == 2 || options.implementation == 3 || options.implementation == 5 || options.useSingles
                        options.SinDelayed = single(options.SinDelayed / options.TOF_bins) + single(options.ScatterC(:));
                    else
                        options.SinDelayed = double(options.SinDelayed / options.TOF_bins) + double(options.ScatterC(:));
                    end
                elseif iscell(options.ScatterC)
                    for kk = 1 : options.partitions
                        if sum(size(options.ScatterC{kk})) > 1
                            if size(options.ScatterC{kk},1) ~= options.Ndist && ~options.use_raw_data
                                options.ScatterC{kk} = permute(options.ScatterC{kk},[2 1 3]);
                            end
                        end
                        if options.normalization_correction && options.normalize_scatter && ~ScatterProp.normalization
                            if isfield(options,'InveonScatter') && isfield(options,'InveonNorm')
                                options.ScatterC{kk} = single(options.ScatterC{kk}) .* reshape(options.normalization, size(options.ScatterC{kk}, 1), size(options.ScatterC{kk}, 2), size(options.ScatterC{kk}, 3));
                            else
                                options.ScatterC{kk} = single(options.ScatterC{kk}) .* reshape(1 ./ options.normalization, size(options.ScatterC{kk}, 1), size(options.ScatterC{kk}, 2), size(options.ScatterC{kk}, 3));
                            end
                        end
                        if options.scatter_variance_reduction && ~ScatterProp.variance_reduction
                            options.ScatterC{kk} = Randoms_variance_reduction(single(options.ScatterC{kk}), options);
                        end
                        if options.scatter_smoothing && ~ScatterProp.smoothing
                            options.ScatterC{kk} = randoms_smoothing(single(options.ScatterC{kk}), options);
                        end
                        if options.use_raw_data == false && options.NSinos ~= options.TotSinos
                            options.ScatterC{kk} = options.ScatterC{kk}(1:options.NSinos*options.Ndist*options.Nang);
                        end
                        if options.implementation == 2 || options.implementation == 3 || options.implementation == 5 || options.useSingles
                            options.SinDelayed = single(options.SinDelayed / options.TOF_bins) + single(options.ScatterC{kk}(:));
                        else
                            options.SinDelayed = double(options.SinDelayed / options.TOF_bins) + double(options.ScatterC{kk}(:));
                        end
                    end
                end
            elseif ~options.scatter_correction && options.TOF_bins > 1
                if options.implementation == 2 || options.implementation == 3 || options.implementation == 5 || options.useSingles
                    options.SinDelayed = single(options.SinDelayed) / single(options.TOF_bins);
                else
                    options.SinDelayed = double(options.SinDelayed) / options.TOF_bins;
                end
            end
        elseif options.scatter_correction
            if sum(size(options.ScatterC)) > 1 && ~iscell(options.ScatterC)
                if size(options.ScatterC,1) ~= options.Ndist && ~options.use_raw_data
                    options.ScatterC = permute(options.ScatterC,[2 1 3]);
                end
                if options.normalization_correction && options.normalize_scatter && ~ScatterProp.normalization
                    if isfield(options,'InveonScatter') && isfield(options,'InveonNorm')
                        options.ScatterC = single(options.ScatterC) .* reshape(options.normalization, size(options.ScatterC, 1), size(options.ScatterC, 2), size(options.ScatterC, 3));
                    else
                        options.ScatterC = single(options.ScatterC) .* reshape(1 ./ options.normalization, size(options.ScatterC, 1), size(options.ScatterC, 2), size(options.ScatterC, 3));
                    end
                end
                if options.scatter_variance_reduction && ~ScatterProp.variance_reduction
                    options.ScatterC = Randoms_variance_reduction(single(options.ScatterC), options);
                end
                if options.scatter_smoothing && ~ScatterProp.smoothing
                    options.ScatterC = randoms_smoothing(single(options.ScatterC), options);
                end
                if options.implementation == 2 || options.implementation == 3 || options.implementation == 5 || options.useSingles
                    options.SinDelayed = single(options.ScatterC(:));
                else
                    options.SinDelayed = double(options.ScatterC(:));
                end
            elseif iscell(options.ScatterC)
                options.SinDelayed = cell(length(options.ScatterC),1);
                for kk = 1 : options.partitions
                    if sum(size(options.ScatterC{1})) > 1
                        if size(options.ScatterC{kk},1) ~= options.Ndist && ~options.use_raw_data
                            options.ScatterC{kk} = permute(options.ScatterC{kk},[2 1 3]);
                        end
                    end
                    if options.normalization_correction && options.normalize_scatter && ~ScatterProp.normalization
                        if isfield(options,'InveonScatter') && isfield(options,'InveonNorm')
                            options.ScatterC{kk} = single(options.ScatterC{kk}) .* reshape(options.normalization, size(options.ScatterC{kk}, 1), size(options.ScatterC{kk}, 2), size(options.ScatterC{kk}, 3));
                        else
                            options.ScatterC{kk} = single(options.ScatterC{kk}) .* reshape(1 ./ options.normalization, size(options.ScatterC{kk}, 1), size(options.ScatterC{kk}, 2), size(options.ScatterC{kk}, 3));
                        end
                    end
                    if options.scatter_variance_reduction && ~ScatterProp.variance_reduction
                        options.ScatterC{kk} = Randoms_variance_reduction(single(options.ScatterC{kk}), options);
                    end
                    if options.scatter_smoothing && ~ScatterProp.smoothing
                        options.ScatterC{kk} = randoms_smoothing(single(options.ScatterC{kk}), options);
                    end
                    if options.implementation == 2 || options.implementation == 3 || options.implementation == 5 || options.useSingles
                        options.SinDelayed{kk} = single(options.ScatterC{kk}(:));
                    else
                        options.SinDelayed{kk} = double(options.ScatterC{kk}(:));
                    end
                end
            end
        end
    end
    if options.sampling > 1 && options.corrections_during_reconstruction
        options.SinDelayed = reshape(options.SinDelayed, options.Ndist, options.Nang, options.NSinos);
        options.SinDelayed = interpolateSinog(options.SinDelayed, options.sampling, options.Ndist, options.partitions, options.sampling_interpolation_method);
        options.SinDelayed = options.SinDelayed(:);
    end
elseif (options.randoms_correction || options.scatter_correction) && ~options.reconstruct_trues && ~options.reconstruct_scatter
    r_exist = isfield(options,'SinDelayed');
    if r_exist && isfield(options,'SinDelayed') && ~iscell(options.SinDelayed) && numel(options.SinDelayed) == 1 && options.randoms_correction
        r_exist = false;
    elseif r_exist && isfield(options,'SinDelayed') && iscell(options.SinDelayed) && numel(options.SinDelayed{1}) == 1 && options.randoms_correction
        r_exist = false;
    end
    options.scatter = false;
    if exist('RandProp','var') == 0 || isempty(RandProp)
        RandProp.smoothing = false;
        RandProp.variance_reduction = false;
    end
    if exist('ScatterProp','var') == 0 || isempty(ScatterProp)
        ScatterProp.smoothing = false;
        ScatterProp.variance_reduction = false;
        ScatterProp.normalization = false;
    end
    if r_exist && options.randoms_correction
        if (~options.variance_reduction && RandProp.variance_reduction) || (~options.randoms_smoothing && RandProp.smoothing)
            %             if options.partitions == 1
            %                 options.SinDelayed = loadStructFromFile(sinoFile,'raw_SinDelayed');
            %             else
            options.SinDelayed = loadStructFromFile(sinoFile, 'raw_SinDelayed');
            %             end
            RandProp.variance_reduction = false;
            RandProp.smoothing = false;
        end
    end
    if options.randoms_correction && ~r_exist
        try
            options.SinDelayed = loadStructFromFile(sinoFile, 'SinDelayed');
        catch
            options = loadDelayedData(options);
        end
    end
    if options.randoms_correction
        [options,~, ~] = applyScatterRandoms(options,RandProp, options.SinDelayed, 0);
    end
    randoms_correction = false;
    options.randoms_correction = false;
    if options.implementation == 2 || options.implementation == 3 || options.implementation == 5 || options.useSingles
        options.SinDelayed = {single(0)};
    else
        options.SinDelayed = {0};
    end
    if options.scatter_correction
        [options,~, ~] = applyScatterRandoms(options,ScatterProp, options.ScatterC, 1);
        options = rmfield(options,'ScatterC');
        options.scatter_correction = false;
    end
    if options.verbose
        disp('Precorrections completed')
    end
else
    randoms_correction = false;
    options.randoms_correction = false;
    options.scatter_correction = false;
    options.scatter = false;
    if options.implementation == 2 || options.implementation == 3 || options.implementation == 5 || options.useSingles
        options.SinDelayed = {single(0)};
    else
        options.SinDelayed = {0};
    end
end

% Load (or compute) normalization correction coefficients
if (options.normalization_correction && options.corrections_during_reconstruction) && ~options.use_user_normalization
    normalization_correction = true;
    if ~isfield(options,'normalization') || isempty(options.normalization)
        if options.use_raw_data
            norm_file = [folder options.machine_name '_normalization_listmode.mat'];
            if exist(norm_file, 'file') == 2
                options.normalization = loadStructFromFile(norm_file,'normalization');
            else
                error('Normalization correction selected, but no normalization data found')
            end
        else
            norm_file = [folder options.machine_name '_normalization_' num2str(options.Ndist) 'x' num2str(options.Nang) '_span' num2str(options.span) '.mat'];
            if exist(norm_file, 'file') == 2
                options.normalization = loadStructFromFile(norm_file,'normalization');
            else
                error('Normalization correction selected, but no normalization data found')
            end
        end
        options.normalization(options.normalization <= 0) = 1;
        options.normalization = 1./options.normalization(:);
        if options.implementation == 2 || options.implementation == 3 || options.implementation == 5 || options.useSingles
            options.normalization = single(options.normalization);
        end
        if ~options.use_raw_data && options.NSinos ~= options.TotSinos
            options.normalization = options.normalization(1 : options.Ndist * options.Nang *options.NSinos);
        end
        if options.sampling > 1 && options.corrections_during_reconstruction
            options.normalization = reshape(options.normalization, options.Ndist, options.Nang, options.NSinos);
            options.normalization = interpolateSinog(options.normalization, options.sampling, options.Ndist, options.partitions, options.sampling_interpolation_method);
            options.normalization = options.normalization(:);
        end
    end
elseif options.normalization_correction && options.use_user_normalization && options.corrections_during_reconstruction
    normalization_correction = true;
    if ~isfield(options,'normalization') || isempty(options.normalization)
        [file, fpath] = uigetfile({'*.nrm;*.mat'},'Select normalization datafile');
        if isequal(file, 0)
            error('No file was selected')
        end
        if any(strfind(file, '.nrm'))
            if options.use_raw_data
                error('Inveon normalization data cannot be used with raw list-mode data')
            end
            fid = fopen(fullfile(fpath, file));
            options.normalization = fread(fid, inf, 'single=>single',0,'l');
            fclose(fid);
            if numel(options.normalization) ~= options.Ndist * options.Nang * options.TotSinos && ~options.use_raw_data
                error('Size mismatch between the current data and the normalization data file')
            end
            options.InveonNorm = true;
        else
            data = load(fullfile(fpath, file));
            variables = fieldnames(data);
            if any(strcmp(variables,'normalization'))
                options.normalization = data.(variables{strcmp(variables,'normalization')});
            else
                options.normalization = data.(variables{1});
            end
            clear data
            if numel(options.normalization) ~= options.Ndist * options.Nang * options.TotSinos && ~options.use_raw_data
                error('Size mismatch between the current data and the normalization data file')
            elseif numel(options.normalization) ~= sum(1:options.detectors) && options.use_raw_data
                error('Size mismatch between the current data and the normalization data file')
            end
        end
        if options.implementation == 2 || options.implementation == 3 || options.implementation == 5 || options.useSingles
            options.normalization = single(options.normalization);
        else
            options.normalization = double(options.normalization);
        end
        options.normalization = 1./options.normalization(:);
        if ~options.use_raw_data && options.NSinos ~= options.TotSinos
            options.normalization = options.normalization(1 : options.Ndist * options.Nang *options.NSinos);
        end
        if options.sampling > 1 && options.corrections_during_reconstruction
            options.normalization = reshape(options.normalization, options.Ndist, options.Nang, options.NSinos);
            options.normalization = interpolateSinog(options.normalization, options.sampling, options.Ndist, options.partitions, options.sampling_interpolation_method);
            options.normalization = options.normalization(:);
        end
    end
elseif options.normalization_correction && ~options.corrections_during_reconstruction
    normalization_correction = false;
    if ~isfield(options,'normalization') || isempty(options.normalization)
        if options.use_user_normalization
            [file, fpath] = uigetfile({'*.nrm;*.mat'},'Select normalization datafile');
            if isequal(file, 0)
                error('No file was selected')
            end
            if any(strfind(file, '.nrm'))
                fid = fopen(fullfile(fpath, file));
                options.normalization = fread(fid, inf, 'single=>single',0,'l');
                fclose(fid);
                options.InveonNorm = true;
            else
                data = load(fullfile(fpath, file));
                variables = fieldnames(data);
                if any(strcmp(variables,'normalization'))
                    options.normalization = data.(variables{strcmp(variables,'normalization')});
                else
                    options.normalization = data.(variables{1});
                end
                clear data
                if iscell(options.SinM)
                    if numel(options.normalization) ~= numel(options.SinM{1})
                        error('Size mismatch between the current data and the normalization data file')
                    end
                else
                    if numel(options.normalization) ~= numel(options.SinM)
                        error('Size mismatch between the current data and the normalization data file')
                    end
                end
            end
            options.normalization = options.normalization(:);
        else
            if options.use_raw_data
                norm_file = [folder options.machine_name '_normalization_listmode.mat'];
            else
                norm_file = [folder options.machine_name '_normalization_' num2str(options.Ndist) 'x' num2str(options.Nang) '_span' num2str(options.span) '.mat'];
            end
            if exist(norm_file, 'file') == 2
                options.normalization = loadStructFromFile(norm_file,'normalization');
            else
                error('Normalization correction selected, but no normalization data found')
            end
            options.normalization = options.normalization(:);
        end
        if ~options.use_raw_data && options.NSinos ~= options.TotSinos
            options.normalization = options.normalization(1 : options.Ndist * options.Nang *options.NSinos);
        end
    end
    NN = options.Ndist * options.Nang * options.NSinos;
    if iscell(options.SinM)
        for kk = 1 : options.partitions
            options.SinM{kk} = options.SinM{kk}(:);
            if options.TOF_bins > 1
                for uu = 1 : options.TOF_bins
                    if options.implementation == 2 || options.implementation == 3 || options.implementation == 5 || options.useSingles
                        options.SinM{kk}(NN * (uu - 1) + 1: NN * uu) = single(full(options.SinM{kk}(NN * (uu - 1) + 1: NN * uu))) .* single(options.normalization);
                    else
                        options.SinM{kk}(NN * (uu - 1) + 1: NN * uu) = full(options.SinM{kk}(NN * (uu - 1) + 1: NN * uu)) .* double(options.normalization);
                    end
                end
            else
                if options.implementation == 2 || options.implementation == 3 || options.implementation == 5 || options.useSingles
                    options.SinM{kk} = single(full(options.SinM{kk})) .* single(options.normalization);
                else
                    options.SinM{kk} = full(options.SinM{kk}) .* double(options.normalization);
                end
            end
        end
    else
        if options.TOF_bins > 1
            options.SinM = options.SinM(:);
            for uu = 1 : options.TOF_bins
                if options.implementation == 2 || options.implementation == 3 || options.implementation == 5 || options.useSingles
                    options.SinM(NN * (uu - 1) + 1: NN * uu) = single(full(options.SinM(NN * (uu - 1) + 1: NN * uu))) .* single(options.normalization);
                else
                    options.SinM(NN * (uu - 1) + 1: NN * uu) = full(options.SinM(NN * (uu - 1) + 1: NN * uu)) .* double(options.normalization);
                end
            end
        else
            options.SinM = options.SinM(:);
            if options.implementation == 2 || options.implementation == 3 || options.implementation == 5 || options.useSingles
                options.SinM = single(full(options.SinM)) .* single(options.normalization);
            else
                options.SinM = full(options.SinM) .* double(options.normalization);
            end
        end
    end
    if options.implementation == 2 || options.implementation == 3 || options.implementation == 5 || options.useSingles
        options.normalization = single(0);
    else
        options.normalization = 0;
    end
    options.normalization_correction = false;
else
    normalization_correction = false;
    if options.implementation == 2 || options.implementation == 3 || options.implementation == 5 || options.useSingles
        options.normalization = single(0);
    else
        options.normalization = 0;
    end
end

if ~isfield(options,'global_correction_factor') || isempty(options.global_correction_factor) || options.global_correction_factor <= 0
    options.global_correction_factor = 1;
end
if ~isfield(options, 'ScatterC')
    if options.implementation == 2 || options.implementation == 3 || options.implementation == 5 || options.useSingles
        options.ScatterC = {single(0)};
    else
        options.ScatterC = 0;
    end
end
if ~isfield(options, 'scatter')
    options.scatter = false;
end
