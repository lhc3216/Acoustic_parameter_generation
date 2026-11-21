function out = run_acoustic_simulation(speed_nii_file, density_nii_file, ...
        kwave_path, output_dir, target_vox, opts)
%RUN_ACOUSTIC_SIMULATION Load paired NIfTI maps and simulate focused ultrasound.
%   out = run_acoustic_simulation(speedFile, densityFile, kwavePath)
%   out = run_acoustic_simulation(speedFile, densityFile, kwavePath, ...
%       outputDir, targetVoxel, opts)
%
% speedFile and densityFile contain physical values in m/s and kg/m^3.
% targetVoxel is an optional one-based [i j k] index in the input volume.
% An empty target uses geometric placement based on the skull mask.
% opts overrides the defaults in simulate_focused_ultrasound. Output filenames
% are set here to acoustic_field.mat and pressure_slice.png under outputDir.
% The default output directory is simulation_outputs/ at the repository root.

if nargin < 3
    error('run_acoustic_simulation:MissingInputs', ...
        'Supply sound speed NIfTI, density NIfTI, and k-Wave toolbox paths.');
end
script_dir = fileparts(mfilename('fullpath'));
if nargin < 4 || isempty(output_dir) || isequal(output_dir, "")
    output_dir = fullfile(script_dir, '..', 'simulation_outputs');
end
if nargin < 5
    target_vox = [];
end
if nargin < 6 || isempty(opts)
    opts = struct();
end
if ~isstruct(opts) || ~isscalar(opts)
    error('run_acoustic_simulation:InvalidOptions', 'opts must be a scalar struct.');
end

validatePathInput(speed_nii_file, 'Sound speed NIfTI');
validatePathInput(density_nii_file, 'Density NIfTI');
validatePathInput(kwave_path, 'k-Wave toolbox');
validatePathInput(output_dir, 'Output directory');
if ~isfile(speed_nii_file)
    error('Sound speed NIfTI does not exist: %s', speed_nii_file);
end
if ~isfile(density_nii_file)
    error('Density NIfTI does not exist: %s', density_nii_file);
end
if ~isfolder(kwave_path)
    error('k-Wave toolbox path does not exist: %s', kwave_path);
end
addpath(genpath(kwave_path));
addpath(script_dir);
if ~exist('kWaveGrid', 'file') || ~exist('kWaveArray', 'file') || ...
        ~exist('kspaceFirstOrder3D', 'file')
    error('The toolbox path must provide kWaveGrid, kWaveArray, and kspaceFirstOrder3D.');
end

speed_info = niftiinfo(speed_nii_file);
density_info = niftiinfo(density_nii_file);

speed_map = single(niftiread(speed_info));
density_map = single(niftiread(density_info));

speed_map = squeeze(speed_map);
density_map = squeeze(density_map);

if ~isequal(size(speed_map), size(density_map))
    error('speed_map and density_map must have the same size. Got [%s] and [%s].', ...
        num2str(size(speed_map)), num2str(size(density_map)));
end

if ndims(speed_map) ~= 3
    error('The loaded acoustic maps must be 3-D arrays.');
end

if any(~isfinite(speed_map(:))) || any(speed_map(:) <= 0)
    error('Sound speed map must contain only positive finite values in m/s.');
end
if any(~isfinite(density_map(:))) || any(density_map(:) <= 0)
    error('Density map must contain only positive finite values in kg/m^3.');
end

voxelSize = niftiVoxelSizeMM(speed_info);
if numel(density_info.PixelDimensions) >= 3
    densityVoxelSize = niftiVoxelSizeMM(density_info);
    if any(abs(voxelSize - densityVoxelSize) > 1e-6)
        error('Speed and density voxel sizes differ after conversion to mm.');
    end
end
validateNiftiRegistration(speed_info, density_info);
warnIfThirdAxisIsNotSuperior(speed_info);

opts.save_file = fullfile(output_dir, 'acoustic_field.mat');
opts.png_file = fullfile(output_dir, 'pressure_slice.png');
opts.input_files = struct('speed', char(speed_nii_file), ...
    'density', char(density_nii_file));

fprintf('Loaded sound speed map: %s\n', speed_nii_file);
fprintf('Loaded density map: %s\n', density_nii_file);
fprintf('Map size: [%s], voxel size: [%.4g %.4g %.4g] mm\n', ...
    num2str(size(speed_map)), voxelSize);
fprintf('Sound speed range: %.3f to %.3f m/s\n', min(speed_map(:)), max(speed_map(:)));
fprintf('Density range: %.3f to %.3f kg/m^3\n', min(density_map(:)), max(density_map(:)));
if isempty(target_vox)
    warning(['target_vox is empty. A geometric fallback target will be used; ' ...
             'set an anatomical target explicitly before quantitative simulation.']);
end

out = simulate_focused_ultrasound(speed_map, density_map, voxelSize, target_vox, opts);

fprintf('Simulation complete.\n');
fprintf('MAT output: %s\n', opts.save_file);
fprintf('PNG output: %s\n', opts.png_file);
end

function validatePathInput(value, label)
is_text_scalar = (ischar(value) && isrow(value)) || ...
    (isstring(value) && isscalar(value));
if ~is_text_scalar || strlength(string(value)) == 0
    error('%s path must be a nonempty character vector or scalar string.', label);
end
end

function voxelSizeMM = niftiVoxelSizeMM(info)
voxelSizeMM = double(info.PixelDimensions(1:3));
units = "";
if isfield(info, 'SpaceUnits') && ~isempty(info.SpaceUnits)
    units = lower(string(info.SpaceUnits));
end

switch units
    case {"millimeter", "millimeters", "mm"}
        scaleToMM = 1;
    case {"meter", "meters", "m"}
        scaleToMM = 1e3;
    case {"micron", "microns", "micrometer", "micrometers", "um"}
        scaleToMM = 1e-3;
    otherwise
        warning('Unknown NIfTI SpaceUnits "%s"; assuming PixelDimensions are in mm.', units);
        scaleToMM = 1;
end
voxelSizeMM = voxelSizeMM * scaleToMM;
end

function validateNiftiRegistration(speed_info, density_info)
Ts = niftiTransformMatrix(speed_info);
Td = niftiTransformMatrix(density_info);
if isempty(Ts) || isempty(Td)
    warning('NIfTI transform metadata is unavailable; speed/density registration was not verified.');
    return;
end

tol = 1e-5 * max(1, max(abs(Ts(:))));
if max(abs(Ts(:) - Td(:))) > tol
    error('Sound speed and density NIfTI affine transforms do not match.');
end
end

function warnIfThirdAxisIsNotSuperior(info)
T = niftiTransformMatrix(info);
if isempty(T)
    return;
end

axis3 = T(3, 1:3);
[~, dominantAxis] = max(abs(axis3));
if dominantAxis ~= 3
    warning(['NIfTI array dimension 3 is not primarily aligned with world z. ' ...
             'The current transducer placement assumes dimension 3 is the head-foot axis.']);
elseif axis3(3) < 0
    warning(['Increasing NIfTI dimension 3 points toward negative world z. ' ...
             'Verify that max(z-index) is the intended transducer side.']);
end
end

function T = niftiTransformMatrix(info)
T = [];
if ~isfield(info, 'Transform') || isempty(info.Transform)
    return;
end
try
    T = double(info.Transform.T);
catch
    if isstruct(info.Transform) && isfield(info.Transform, 'T')
        T = double(info.Transform.T);
    end
end
if ~isequal(size(T), [4 4]) || any(~isfinite(T(:)))
    T = [];
end
end
