function out = simulate_focused_ultrasound(speed_map, density_map, voxelSize, target_vox, opts)
%SIMULATE_FOCUSED_ULTRASOUND Simulate a focused bowl in a 3-D acoustic medium.
% The 3-D solver records the X-Z plane through the target, not the full volume.
%
% Required inputs:
%   speed_map   : [Nx Ny Nz] sound speed map, m/s
%   density_map : [Nx Ny Nz] density map, kg/m^3
%
% Optional inputs:
%   voxelSize   : [dx dy dz], mm. Default is [0.5 0.5 0.5].
%   target_vox  : focal voxel [ix iy iz]. If omitted, opts.target_mode
%                 selects an outward center-y ray through the skull.
%   opts        : optional parameters, see defaultSimulationOptions().
%
% Outputs:
%   out         : pressure-plane data, source geometry, and convergence metadata.
% Results are also saved to opts.save_file; opts.png_file selects the figure.

if nargin < 3 || isempty(voxelSize)
    voxelSize = [0.5 0.5 0.5];
end
if nargin < 4
    target_vox = [];
end
if nargin < 5 || isempty(opts)
    opts = struct();
end
if isstruct(voxelSize)
    opts = voxelSize;
    voxelSize = [0.5 0.5 0.5];
    target_vox = [];
elseif isstruct(target_vox)
    opts = target_vox;
    target_vox = [];
end
  
opts = validateSimulationOptions(resolveSimulationOptions(opts));
voxelSize = double(voxelSize(:)).';
if numel(voxelSize) ~= 3 || any(voxelSize <= 0) || any(~isfinite(voxelSize))
    error('voxelSize must be [dx dy dz] in mm.');
end
if max(abs(voxelSize - voxelSize(1))) > 1e-6 * voxelSize(1)
    error('kWaveArray requires an isotropic grid; dx, dy, and dz must match.');
end
if ~isequal(size(speed_map), size(density_map))
    error('speed_map and density_map must have the same size.');
end
if ndims(speed_map) ~= 3
    error('speed_map and density_map must be 3-D arrays.');
end
if any(~isfinite(speed_map(:))) || any(speed_map(:) <= 0)
    error('speed_map must contain only positive finite values in m/s.');
end
if any(~isfinite(density_map(:))) || any(density_map(:) <= 0)
    error('density_map must contain only positive finite values in kg/m^3.');
end

input_grid_size = size(speed_map);
target_vox_input = target_vox;
if ~isempty(target_vox_input)
    target_vox_input = round(double(target_vox_input(:)).');
    if numel(target_vox_input) ~= 3 || any(target_vox_input < 1) || ...
            any(target_vox_input > input_grid_size)
        error('target_vox must be inside the input (unpadded) acoustic maps.');
    end
end

dx = voxelSize(1) * 1e-3;
dy = voxelSize(2) * 1e-3;
dz = voxelSize(3) * 1e-3;
% Add water padding while preserving the input target coordinates.
[speed_map, density_map, z_prepad] = prepareMediumZBounds(speed_map, density_map, opts);
if ~isempty(target_vox_input)
    target_vox = target_vox_input + [0 0 z_prepad];
end
[Nx, Ny, Nz] = size(speed_map);
if any([Nx Ny Nz] <= 2 * opts.PMLSize + 4)
    error('PMLSize is too large for the padded simulation grid.');
end

skull_mask = makeSkullMask(speed_map, density_map, opts);
assertSkullOutsidePML(skull_mask, opts.PMLSize);

geometry_opts = fixedGeometryInSimulationCoords(opts, z_prepad);
[target_vox, target_geometry] = chooseTargetVoxel( ...
    target_vox, skull_mask, [Nx Ny Nz], voxelSize, geometry_opts);
target_vox_unpadded = target_vox - [0 0 z_prepad];

%% Define acoustic properties
medium.sound_speed = speed_map;
medium.density = density_map;
if opts.skull_alpha_coeff > 0
    medium.alpha_coeff = zeros(size(speed_map), 'like', speed_map);
    medium.alpha_coeff(skull_mask) = opts.skull_alpha_coeff;
    medium.alpha_power = opts.alpha_power;
else
    warning(['Skull absorption is disabled (opts.skull_alpha_coeff = 0). ' ...
             'Pressure transmission is lossless and may be overestimated.']);
end

%% Build the simulation grid and time step
kgrid = kWaveGrid(Nx, dx, Ny, dy, Nz, dz);
cmax = double(max(speed_map(:)));
cmin = double(min(speed_map(:)));
if cmin <= 0 || ~isfinite(cmin) || ~isfinite(cmax)
    error('speed_map must contain positive finite values.');
end
kgrid.makeTime(cmax, opts.CFL);
dt = kgrid.dt;

%% Record the target X-Z plane
sensor = createPressurePlaneSensor([Nx Ny Nz], target_vox, opts.PMLSize);
[sensor_vox, locS, local_size, local_axes_mm] = getSensorCoordinates(sensor.mask, voxelSize);

%% Place and drive the focused bowl source
[source, source_signal, array_mask, source_info] = createFocusedBowlSource( ...
    kgrid, skull_mask, target_geometry, opts);

[Nt, simulation_timing] = estimateTimeSteps( ...
    kgrid, sensor_vox, array_mask, cmin, opts);
kgrid.setTime(Nt, dt);
t = double(kgrid.t_array(:)).';
simulation_timing.dt_s = dt;
simulation_timing.actual_end_time_s = double(t(end));
source_signal = makeRampSignal(t, opts.f0, opts.source_amp, opts.ramp_cycles);
source.p = source_info.karray.getDistributedSourceSignal(kgrid, source_signal);

input_args = {'PMLSize', opts.PMLSize, 'PMLInside', true, ...
    'Smooth', false, 'PlotSim', false, 'PlotPML', false, ...
    'DisplayMask', 'off', 'DataCast', opts.DataCast, 'DataRecast', true};

fprintf('Running single focused bowl: f0 = %.3f MHz, amp = %.3g Pa.\n', ...
    opts.f0 / 1e6, opts.source_amp);
fprintf('Bowl position voxel approx [%d %d %d], focus voxel [%d %d %d].\n', ...
    source_info.bowl_vox_approx(1), source_info.bowl_vox_approx(2), ...
    source_info.bowl_vox_approx(3), target_vox(1), target_vox(2), target_vox(3));
fprintf('ROC = %.1f mm, diameter = %.1f mm, source voxels = %d.\n', ...
    source_info.roc_m * 1e3, source_info.diameter_m * 1e3, nnz(array_mask));
if isfinite(source_info.physical_gap_m)
    fprintf('Requested bowl-surface axial skull gap = %.2f mm.\n', ...
        source_info.physical_gap_m * 1e3);
end

sensor_data = runKWaveSolver(kgrid, medium, source, sensor, input_args);

%% Extract the steady-state field at the source frequency
[P_f0_3d, I_Wm2, p_vol_t, steady_state] = extractSteadyStateField( ...
    sensor_data, locS, local_size, t, dt, opts.f0, speed_map, density_map, ...
    sensor_vox, array_mask, opts);
clear sensor_data;
if ~opts.save_time_series
    p_vol_t = [];
end

%% Summarize and save the pressure plane
[field_peak_vox, field_peak_pressure_pa, target_pressure_pa, focus_offset_mm] = ...
    summarizePressureField(P_f0_3d, array_mask, sensor_vox, target_vox, voxelSize);
fprintf('Target pressure = %.4g Pa; non-source peak = %.4g Pa at voxel [%d %d %d].\n', ...
    target_pressure_pa, field_peak_pressure_pa, ...
    field_peak_vox(1), field_peak_vox(2), field_peak_vox(3));
fprintf('Peak offset from target = [%.2f %.2f %.2f] mm (distance %.2f mm).\n', ...
    focus_offset_mm(1), focus_offset_mm(2), focus_offset_mm(3), ...
    norm(focus_offset_mm));

out = struct();
out.P_f0_3d = P_f0_3d;
out.I_Wm2 = I_Wm2;
if opts.save_time_series
    out.p_vol_t = p_vol_t;
end
out.array_mask = array_mask;
out.sensor_mask = sensor.mask;
out.steady_state = steady_state;
out.simulation_timing = simulation_timing;
out.target_vox = target_vox;
out.target_vox_input = target_vox_input;
out.target_vox_unpadded = target_vox_unpadded;
out.field_peak_vox = field_peak_vox;
out.field_peak_pressure_pa = field_peak_pressure_pa;
out.target_pressure_pa = target_pressure_pa;
out.focus_offset_mm = focus_offset_mm;
out.z_prepad = z_prepad;
out.sensor_vox = sensor_vox;
out.local_size = local_size;
out.local_axes_mm = local_axes_mm;
out.source_info = stripSourceInfo(source_info);
out.voxelSize = voxelSize;
out.dt = dt;
out.t = single(t);
out.opts = opts;
out.f0 = opts.f0;
out.png_file = opts.png_file;
out.intensity_is_plane_wave_approx = true;

savePressureSlice(abs(P_f0_3d), skull_mask, array_mask, target_vox, ...
    source_info, local_axes_mm, sensor_vox, voxelSize, opts);

save_dir = fileparts(opts.save_file);
if ~isempty(save_dir) && ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

save(opts.save_file, '-struct', 'out', '-v7.3');
fprintf('Saved simulation field data to %s\n', opts.save_file);

end

function opts = resolveSimulationOptions(opts)
defaults = defaultSimulationOptions();
names = fieldnames(defaults);
for ii = 1:numel(names)
    name = names{ii};
    if ~isfield(opts, name) || isempty(opts.(name))
        opts.(name) = defaults.(name);
    end
end
end

function opts = defaultSimulationOptions()
opts = struct();
opts.target_mode = -1; % 0 -> normal, -1 -> CCW 30 deg, 1 -> CW 30 deg
opts.c_water = 1482.3;
opts.rho_water = 1000;
opts.f0 = 550e3;
opts.source_amp = 1e5;
opts.source_roc_mm = 58;
opts.source_diameter_mm = 60;
opts.source_gap_mm = 15;
opts.CFL = 0.3;
opts.PMLSize = 16;
opts.DataCast = 'gpuArray-single';
opts.ramp_cycles = 2;
opts.steady_cycles = 9;
opts.extra_cycles = 3;
opts.nCycles_win = 6;
opts.postprocess_version = 3;
opts.steady_state_tolerance = 0.03;
opts.steady_state_settle_cycles = 24;
opts.steady_state_check_windows = 3;
opts.steady_state_check_floor_db = -40;
opts.postprocess_chunk_points = 2048;
opts.bli_tolerance = 0.1;
opts.upsampling_rate = 4;
opts.speed_threshold = 1800;
opts.density_threshold = 1200;
opts.skull_alpha_coeff = 0;
opts.alpha_power = 1.5;
opts.save_time_series = false;
opts.visualize_forward_only = true;
opts.visualize_z_margin_mm = 2;
opts.visualize_db_floor = -40;
opts.z_extend_vox = 352;
opts.save_file = 'acoustic_field.mat';
opts.png_file = 'pressure_slice.png';
end

function [speed_map, density_map, bottom_n] = prepareMediumZBounds(speed_map, density_map, opts)
target_Nz = round(opts.z_extend_vox);
if ~isscalar(target_Nz) || target_Nz < 1 || ~isfinite(target_Nz)
    error('opts.z_extend_vox must be a positive scalar voxel count.');
end

bottom_n = round(opts.PMLSize);
if ~isscalar(bottom_n) || bottom_n < 0 || ~isfinite(bottom_n)
    error('opts.PMLSize must be a non-negative integer voxel count.');
end
if bottom_n > 0
    pad_size = [size(speed_map, 1), size(speed_map, 2), bottom_n];
    speed_bottom = repmat(cast(opts.c_water, 'like', speed_map), pad_size);
    density_bottom = repmat(cast(opts.rho_water, 'like', density_map), pad_size);
    speed_map = cat(3, speed_bottom, speed_map);
    density_map = cat(3, density_bottom, density_map);
end

old_Nz = size(speed_map, 3);
if old_Nz < target_Nz
    speed_map(:,:,old_Nz + 1:target_Nz) = opts.c_water;
    density_map(:,:,old_Nz + 1:target_Nz) = opts.rho_water;
elseif old_Nz > target_Nz
    warning('Current Nz (%d) is greater than opts.z_extend_vox (%d); no z cropping applied.', ...
        old_Nz, target_Nz);
end

fprintf('Prepared medium z bounds: prepended %d water voxels, final Nz = %d.\n', ...
    bottom_n, size(speed_map, 3));
end

function opts_out = fixedGeometryInSimulationCoords(opts_in, z_prepad)
opts_out = opts_in;
if ~isfield(opts_in, 'fixed_source_geometry') || isempty(opts_in.fixed_source_geometry)
    return;
end

fixed = opts_in.fixed_source_geometry;
if ~isstruct(fixed) || ~isscalar(fixed)
    error('opts.fixed_source_geometry must be a scalar struct.');
end
if ~isfield(fixed, 'coordinate_space')
    error(['opts.fixed_source_geometry.coordinate_space must be ''input'' ' ...
           'or ''simulation''.']);
end

coordinate_space = lower(string(fixed.coordinate_space));
if coordinate_space == "input" || coordinate_space == "unpadded"
    shift = [0 0 z_prepad];
    coordinate_fields = {'focus_vox_continuous', 'bowl_vox_continuous', ...
        'skull_hit_vox', 'ray_origin_vox'};
    for ii = 1:numel(coordinate_fields)
        name = coordinate_fields{ii};
        if isfield(fixed, name)
            coordinate_value = fixed.(name);
        else
            coordinate_value = [];
        end
        if ~isempty(coordinate_value) && all(isfinite(double(coordinate_value(:))))
            fixed.(name) = double(coordinate_value(:)).' + shift;
        end
    end
    fixed.coordinate_space = 'simulation';
elseif coordinate_space ~= "simulation"
    error(['opts.fixed_source_geometry.coordinate_space must be ''input'' ' ...
           'or ''simulation''.']);
end
opts_out.fixed_source_geometry = fixed;
end

function skull_mask = makeSkullMask(speed_map, density_map, opts)
skull_mask = speed_map >= opts.speed_threshold | ...
    density_map >= opts.density_threshold;
if ~any(skull_mask(:))
    warning('No skull voxels found by thresholds; skull_mask is empty.');
else
    skull_fraction = nnz(skull_mask) / numel(skull_mask);
    fprintf('Skull mask: %d voxels (%.2f%% of grid).\n', ...
        nnz(skull_mask), 100 * skull_fraction);
    if skull_fraction > 0.30
        warning(['Skull mask occupies more than 30%% of the grid. Check that ' ...
                 'soft tissue is not being classified as skull.']);
    end
end
end

function assertSkullOutsidePML(skull_mask, pml_size)
if pml_size == 0 || ~any(skull_mask(:))
    return;
end
[Nx, Ny, Nz] = size(skull_mask);
interior = false(size(skull_mask));
interior((pml_size + 1):(Nx - pml_size), ...
         (pml_size + 1):(Ny - pml_size), ...
         (pml_size + 1):(Nz - pml_size)) = true;
skull_in_pml = skull_mask & ~interior;
if any(skull_in_pml(:))
    error(['The skull mask reaches the internal PML at %d grid points. ' ...
           'Add water padding or reduce PMLSize before simulation.'], ...
          nnz(skull_in_pml));
end
end

function opts = validateSimulationOptions(opts)
positive_names = {'c_water', 'rho_water', 'f0', 'source_amp', ...
    'source_roc_mm', 'source_diameter_mm', 'source_gap_mm', 'CFL', ...
    'nCycles_win', 'z_extend_vox', 'steady_state_tolerance', ...
    'steady_state_check_windows', 'postprocess_chunk_points'};
for ii = 1:numel(positive_names)
    name = positive_names{ii};
    value = opts.(name);
    if ~isscalar(value) || ~isfinite(value) || value <= 0
        error('opts.%s must be a positive finite scalar.', name);
    end
end

nonnegative_names = {'ramp_cycles', 'steady_cycles', 'extra_cycles', ...
    'skull_alpha_coeff', 'steady_state_settle_cycles'};
for ii = 1:numel(nonnegative_names)
    name = nonnegative_names{ii};
    value = opts.(name);
    if ~isscalar(value) || ~isfinite(value) || value < 0
        error('opts.%s must be a non-negative finite scalar.', name);
    end
end

opts.PMLSize = round(opts.PMLSize);
if ~isscalar(opts.PMLSize) || ~isfinite(opts.PMLSize) || opts.PMLSize < 0
    error('opts.PMLSize must be a non-negative integer voxel count.');
end
if opts.steady_cycles < opts.nCycles_win
    error('opts.steady_cycles must be at least opts.nCycles_win.');
end
opts.postprocess_chunk_points = round(opts.postprocess_chunk_points);
if opts.postprocess_chunk_points < 1
    error('opts.postprocess_chunk_points must round to at least 1.');
end
if ~isscalar(opts.postprocess_version) || ~isfinite(opts.postprocess_version) || ...
        opts.postprocess_version < 1 || opts.postprocess_version ~= round(opts.postprocess_version)
    error('opts.postprocess_version must be a positive integer.');
end
if opts.steady_state_tolerance > 1
    error('opts.steady_state_tolerance must not exceed 1.');
end
if opts.steady_state_check_windows ~= round(opts.steady_state_check_windows) || ...
        opts.steady_state_check_windows < 3
    error('opts.steady_state_check_windows must be an integer of at least 3.');
end
if ~isscalar(opts.steady_state_check_floor_db) || ...
        ~isfinite(opts.steady_state_check_floor_db) || ...
        opts.steady_state_check_floor_db >= 0
    error('opts.steady_state_check_floor_db must be a negative finite scalar.');
end
if ~isscalar(opts.target_mode) || ~isfinite(opts.target_mode) || ...
        ~ismember(opts.target_mode, [-1 0 1])
    error('opts.target_mode must be -1, 0, or 1.');
end
if opts.source_diameter_mm >= 2 * opts.source_roc_mm
    error('opts.source_diameter_mm must be smaller than twice opts.source_roc_mm.');
end
if ~isscalar(opts.bli_tolerance) || opts.bli_tolerance <= 0 || opts.bli_tolerance > 1
    error('opts.bli_tolerance must be in (0, 1].');
end
if ~isscalar(opts.upsampling_rate) || opts.upsampling_rate <= 0
    error('opts.upsampling_rate must be positive.');
end
if ~isscalar(opts.alpha_power) || opts.alpha_power <= 0 || opts.alpha_power >= 3
    error('opts.alpha_power must be between 0 and 3.');
end
if ~islogical(opts.save_time_series) || ~isscalar(opts.save_time_series)
    error('opts.save_time_series must be a logical scalar.');
end
if ~islogical(opts.visualize_forward_only) || ~isscalar(opts.visualize_forward_only)
    error('opts.visualize_forward_only must be a logical scalar.');
end
if ~isscalar(opts.visualize_z_margin_mm) || opts.visualize_z_margin_mm < 0
    error('opts.visualize_z_margin_mm must be a non-negative scalar.');
end
if ~isscalar(opts.visualize_db_floor) || opts.visualize_db_floor >= 0
    error('opts.visualize_db_floor must be a negative scalar.');
end
end

function [target_vox, geometry] = chooseTargetVoxel( ...
        target_vox, skull_mask, grid_size, voxelSize, opts)
% The ray points outward: center_bottom -> focus -> skull -> bowl.
ray_direction = targetModeRayDirection(opts.target_mode);
incident_direction = -ray_direction;
half_diameter_mm = opts.source_diameter_mm / 2;
sag_mm = opts.source_roc_mm - sqrt( ...
    opts.source_roc_mm^2 - half_diameter_mm^2);
geometry = struct();
geometry.ray_direction = ray_direction;
geometry.incident_direction = incident_direction;
geometry.sag_mm = sag_mm;

if isfield(opts, 'fixed_source_geometry') && ~isempty(opts.fixed_source_geometry)
    if ~isempty(target_vox)
        error('target_vox must be empty when opts.fixed_source_geometry is supplied.');
    end
    fixed = opts.fixed_source_geometry;
    required = {'focus_vox_continuous', 'bowl_vox_continuous'};
    for ii = 1:numel(required)
        name = required{ii};
        if ~isfield(fixed, name)
            error('opts.fixed_source_geometry.%s is required.', name);
        end
        fixed_value = fixed.(name);
        value = double(fixed_value(:)).';
        if numel(value) ~= 3 || any(~isfinite(value))
            error('opts.fixed_source_geometry.%s must be a finite 1-by-3 vector.', name);
        end
        geometry.(name) = value;
    end

    source_axis_mm = (geometry.bowl_vox_continuous - ...
        geometry.focus_vox_continuous) .* voxelSize;
    source_roc_mm = norm(source_axis_mm);
    roc_tolerance_mm = 1e-6 * max(1, opts.source_roc_mm);
    if source_roc_mm <= 0 || abs(source_roc_mm - opts.source_roc_mm) > roc_tolerance_mm
        error(['Fixed focus-to-bowl distance is %.6g mm, but ' ...
               'opts.source_roc_mm is %.6g mm.'], source_roc_mm, opts.source_roc_mm);
    end
    geometry.ray_direction = source_axis_mm ./ source_roc_mm;
    geometry.incident_direction = -geometry.ray_direction;
    geometry.is_automatic = false;
    geometry.skull_hit_vox = [NaN NaN NaN];
    geometry.ray_origin_vox = [NaN NaN NaN];
    geometry.rear_center_gap_mm = NaN;
    target_vox = round(geometry.focus_vox_continuous);
    fprintf(['Using fixed source geometry: bowl center [%.3f %.3f %.3f], ' ...
             'continuous focus [%.3f %.3f %.3f].\n'], ...
        geometry.bowl_vox_continuous(1), geometry.bowl_vox_continuous(2), ...
        geometry.bowl_vox_continuous(3), geometry.focus_vox_continuous(1), ...
        geometry.focus_vox_continuous(2), geometry.focus_vox_continuous(3));
elseif ~isempty(target_vox)
    target_vox = round(double(target_vox(:)).');
    if numel(target_vox) ~= 3 || any(target_vox < 1) || any(target_vox > grid_size)
        error('target_vox must be inside the simulation grid.');
    end
    geometry.is_automatic = false;
    geometry.focus_vox_continuous = double(target_vox);
    roc_vox = opts.source_roc_mm .* ray_direction ./ voxelSize;
    geometry.bowl_vox_continuous = double(target_vox) + roc_vox;
    geometry.skull_hit_vox = [NaN NaN NaN];
    geometry.ray_origin_vox = [NaN NaN NaN];
    geometry.rear_center_gap_mm = NaN;
else
    Nx = grid_size(1);
    Ny = grid_size(2);
    Nz = grid_size(3);
    if any(skull_mask(:))
        [ix, iy, ~] = ind2sub(grid_size, find(skull_mask));
        center_x = round(median(ix));
        center_y = round(median(iy));
        center_bottom = findCranialRayOrigin( ...
            skull_mask, center_x, center_y, opts.PMLSize);
        skull_hit_vox = outerSkullVoxelOnRay( ...
            skull_mask, center_bottom, ray_direction, voxelSize, opts.target_mode);

        rear_center_gap_mm = opts.source_gap_mm + sag_mm;
        gap_vox = rear_center_gap_mm .* ray_direction ./ voxelSize;
        roc_vox = opts.source_roc_mm .* ray_direction ./ voxelSize;
        bowl_vox_continuous = double(skull_hit_vox) + gap_vox;
        focus_vox_continuous = bowl_vox_continuous - roc_vox;
        target_vox = round(focus_vox_continuous);

        hit_distance_mm = dot( ...
            (double(skull_hit_vox) - center_bottom) .* voxelSize, ray_direction);
        focus_distance_mm = dot( ...
            (focus_vox_continuous - center_bottom) .* voxelSize, ray_direction);
        if focus_distance_mm <= 0 || focus_distance_mm >= hit_distance_mm
            error(['The selected ROC/gap does not produce the required order ' ...
                   'center_bottom -> focus -> skull -> bowl.']);
        end

        geometry.is_automatic = true;
        geometry.ray_origin_vox = center_bottom;
        geometry.skull_hit_vox = skull_hit_vox;
        geometry.bowl_vox_continuous = bowl_vox_continuous;
        geometry.focus_vox_continuous = focus_vox_continuous;
        geometry.rear_center_gap_mm = rear_center_gap_mm;
        warning(['Using target_mode %d: outer skull hit [%d %d %d], bowl center ' ...
                 '[%.2f %.2f %.2f], target voxel [%d %d %d].'], ...
            opts.target_mode, ...
            skull_hit_vox(1), skull_hit_vox(2), skull_hit_vox(3), ...
            bowl_vox_continuous(1), bowl_vox_continuous(2), bowl_vox_continuous(3), ...
            target_vox(1), target_vox(2), target_vox(3));
    else
        target_vox = [round(Nx / 2), round(Ny / 2), round(Nz / 2)];
        geometry.is_automatic = false;
        geometry.focus_vox_continuous = double(target_vox);
        roc_vox = opts.source_roc_mm .* ray_direction ./ voxelSize;
        geometry.bowl_vox_continuous = double(target_vox) + roc_vox;
        geometry.skull_hit_vox = [NaN NaN NaN];
        geometry.ray_origin_vox = [NaN NaN NaN];
        geometry.rear_center_gap_mm = NaN;
        warning('Using grid-center target because skull_mask is empty.');
    end
end

interior_min = [1 1 1] + opts.PMLSize;
interior_max = grid_size - opts.PMLSize;
if any(target_vox < 1) || any(target_vox > grid_size)
    error(['The target selected by opts.target_mode lies outside the grid. ' ...
           'Increase the grid padding or reduce opts.source_roc_mm.']);
end
if any(target_vox < interior_min) || any(target_vox > interior_max)
    error('Target voxel lies inside the PML. Move the target into the interior domain.');
end
if any(geometry.focus_vox_continuous < interior_min) || ...
        any(geometry.focus_vox_continuous > interior_max)
    error('The continuous focus lies inside the PML or outside the grid.');
end
if any(geometry.bowl_vox_continuous < interior_min) || ...
        any(geometry.bowl_vox_continuous > interior_max)
    error(['The bowl center lies inside the PML or outside the grid. ' ...
           'Increase grid padding or reduce the requested source geometry.']);
end
if skull_mask(target_vox(1), target_vox(2), target_vox(3))
    error('Target voxel is classified as skull. Check target_vox or skull thresholds.');
end
end

function direction = targetModeRayDirection(target_mode)
angle_deg = 30 * target_mode;
direction = [sind(angle_deg), 0, cosd(angle_deg)];
direction = direction ./ norm(direction);
end

function origin_vox = findCranialRayOrigin( ...
        skull_mask, center_x, center_y, pml_size)
grid_size = size(skull_mask);
interior_min = [1 1 1] + pml_size;
interior_max = grid_size - pml_size;
origin_vox = [center_x, center_y, interior_min(3)];
if any(origin_vox < interior_min) || any(origin_vox > interior_max) || ...
        skull_mask(origin_vox(1), origin_vox(2), origin_vox(3))
    error(['The center of the cropped bottom slice is not a valid non-skull ' ...
           'ray origin. Check skull_mask or provide target_vox explicitly.']);
end
end

function hit_vox = outerSkullVoxelOnRay( ...
        skull_mask, ray_origin_vox, direction, voxelSize, target_mode)
grid_size = size(skull_mask);
origin_mm = (double(ray_origin_vox) - 1) .* voxelSize;
corner_span_mm = (double(grid_size) - 1) .* voxelSize;
max_distance_mm = norm(corner_span_mm) + norm(voxelSize);
step_mm = min(voxelSize) / 4;
distance_mm = (0:step_mm:max_distance_mm).';
ray_mm = origin_mm + distance_mm .* direction;
ray_vox = round(ray_mm ./ voxelSize) + 1;
inside = all(ray_vox >= 1 & ray_vox <= grid_size, 2);
ray_vox = ray_vox(inside, :);
if isempty(ray_vox)
    error('The target ray does not enter the simulation grid.');
end

ray_vox = unique(ray_vox, 'rows', 'stable');
ray_linear = sub2ind(grid_size, ray_vox(:,1), ray_vox(:,2), ray_vox(:,3));
on_skull = skull_mask(ray_linear);
first_hit = find(on_skull, 1, 'first');
if isempty(first_hit)
    error(['The target_mode %d ray does not intersect skull_mask. ' ...
           'Check the skull mask or choose another incidence mode.'], target_mode);
end

first_exit = find(~on_skull((first_hit + 1):end), 1, 'first');
if isempty(first_exit)
    outer_hit = numel(on_skull);
else
    outer_hit = first_hit + first_exit - 1;
end
hit_vox = ray_vox(outer_hit, :);
end

function sensor = createPressurePlaneSensor(grid_size, target_vox, pml_size)
ix = (pml_size + 1):(grid_size(1) - pml_size);
iy = target_vox(2);
iz = (pml_size + 1):(grid_size(3) - pml_size);
if isempty(ix) || isempty(iz) || iy <= pml_size || iy > grid_size(2) - pml_size
    error('PMLSize leaves no valid center-y sensor plane.');
end
[Xv, Yv, Zv] = ndgrid(ix, iy, iz);
sensor.mask = false(grid_size);
sensor.mask(sub2ind(grid_size, Xv(:), Yv(:), Zv(:))) = true;
sensor.record = {'p'};
end

function [sensor_vox, locS, local_size, local_axes_mm] = getSensorCoordinates(sensor_mask, voxelSize)
grid_size = size(sensor_mask);
[sx, sy, sz] = ind2sub(grid_size, find(sensor_mask));
sensor_find = [sx, sy, sz];

ix = unique(sx).';
iy = unique(sy).';
iz = unique(sz).';
[Xv, Yv, Zv] = ndgrid(ix, iy, iz);
sensor_vox = [Xv(:), Yv(:), Zv(:)];
local_size = [numel(ix), numel(iy), numel(iz)];

lin_order = sub2ind(grid_size, sensor_vox(:,1), sensor_vox(:,2), sensor_vox(:,3));
lin_find = sub2ind(grid_size, sensor_find(:,1), sensor_find(:,2), sensor_find(:,3));
[tfS, locS] = ismember(lin_order, lin_find);
if ~all(tfS)
    error('Sensor order mapping failed.');
end

local_axes_mm = { ...
    (ix - 1) * voxelSize(1), ...
    (iy - 1) * voxelSize(2), ...
    (iz - 1) * voxelSize(3)};
end

function [source, source_signal, array_mask, info] = createFocusedBowlSource( ...
    kgrid, skull_mask, target_geometry, opts)
[Nx, Ny, Nz] = size(skull_mask);
focus_pos = voxelPositionToGrid(kgrid, target_geometry.focus_vox_continuous);
bowl_pos = voxelPositionToGrid(kgrid, target_geometry.bowl_vox_continuous);
roc_m = opts.source_roc_mm * 1e-3;
if roc_m <= 0
    error('opts.source_roc_mm must be positive.');
end

pml_size = opts.PMLSize;
diameter_m = opts.source_diameter_mm * 1e-3;
if diameter_m <= 2 * max(kgrid.dx, kgrid.dy)
    error('opts.source_diameter_mm is too small for the current grid spacing.');
end

sag_m = roc_m - sqrt(max(roc_m^2 - (diameter_m / 2)^2, 0));
if target_geometry.is_automatic
    physical_gap_m = opts.source_gap_mm * 1e-3;
    rear_center_gap_m = target_geometry.rear_center_gap_mm * 1e-3;
else
    physical_gap_m = NaN;
    rear_center_gap_m = NaN;
end
assertCompleteBowlInsideGrid( ...
    kgrid, bowl_pos, focus_pos, roc_m, diameter_m, pml_size);

karray = kWaveArray('BLITolerance', opts.bli_tolerance, ...
    'UpsamplingRate', opts.upsampling_rate, 'SinglePrecision', true);
karray.addBowlElement(bowl_pos, roc_m, diameter_m, focus_pos);

source = struct();
source.p_mask = karray.getArrayBinaryMask(kgrid);
array_mask = source.p_mask;
if ~any(array_mask(:))
    error('Focused bowl source mask is empty. Check source position and grid size.');
end

interior_mask = false(size(skull_mask));
interior_mask((pml_size + 1):(Nx - pml_size), ...
              (pml_size + 1):(Ny - pml_size), ...
              (pml_size + 1):(Nz - pml_size)) = true;
pml_overlap = array_mask & ~interior_mask;
if any(pml_overlap(:))
    error(['The distributed bowl source reaches the PML at %d grid points. ' ...
           'Increase padding or BLITolerance, or reduce the aperture.'], ...
          nnz(pml_overlap));
end

numerical_overlap = array_mask & skull_mask;
if any(numerical_overlap(:))
    bli_half_width_mm = ceil(1 / (pi * opts.bli_tolerance)) * kgrid.dx * 1e3;
    error(['The non-local BLI source support overlaps the skull at %d grid points ' ...
           '(approximate BLI half-width %.2f mm). Increase opts.bli_tolerance ' ...
           'or opts.source_gap_mm.'], nnz(numerical_overlap), bli_half_width_mm);
end
source_signal = [];

info = struct();
info.karray = karray;
info.bowl_pos_m = bowl_pos;
info.focus_pos_m = focus_pos;
info.roc_m = roc_m;
info.diameter_m = diameter_m;
info.sag_m = sag_m;
info.physical_gap_m = physical_gap_m;
info.rear_center_gap_m = rear_center_gap_m;
info.ray_direction = target_geometry.ray_direction;
info.incident_direction = target_geometry.incident_direction;
info.skull_hit_vox = target_geometry.skull_hit_vox;
info.ray_origin_vox = target_geometry.ray_origin_vox;
info.focus_vox_continuous = target_geometry.focus_vox_continuous;
info.bowl_vox_continuous = target_geometry.bowl_vox_continuous;
info.bowl_vox_approx = [nearestIndex(kgrid.x_vec, bowl_pos(1)), ...
    nearestIndex(kgrid.y_vec, bowl_pos(2)), nearestIndex(kgrid.z_vec, bowl_pos(3))];
end

function position_m = voxelPositionToGrid(kgrid, position_vox)
position_m = [double(kgrid.x_vec(1)) + (position_vox(1) - 1) * kgrid.dx, ...
              double(kgrid.y_vec(1)) + (position_vox(2) - 1) * kgrid.dy, ...
              double(kgrid.z_vec(1)) + (position_vox(3) - 1) * kgrid.dz];
end

function assertCompleteBowlInsideGrid( ...
        kgrid, bowl_pos, focus_pos, roc_m, diameter_m, pml_size)
axis_direction = focus_pos - bowl_pos;
axis_direction = axis_direction ./ norm(axis_direction);
if abs(axis_direction(3)) < 0.9
    reference = [0 0 1];
else
    reference = [0 1 0];
end
basis_u = cross(axis_direction, reference);
basis_u = basis_u ./ norm(basis_u);
basis_v = cross(axis_direction, basis_u);

min_spacing = min([kgrid.dx, kgrid.dy, kgrid.dz]);
radial_count = max(8, ceil((diameter_m / 2) / min_spacing) + 1);
radial_m = linspace(0, diameter_m / 2, radial_count);
theta = linspace(0, 2 * pi, 181);
surface_min = [Inf Inf Inf];
surface_max = [-Inf -Inf -Inf];
for rr = radial_m
    cap_depth = roc_m - sqrt(max(roc_m^2 - rr^2, 0));
    ring = bowl_pos + cap_depth .* axis_direction + ...
        rr .* cos(theta(:)) .* basis_u + rr .* sin(theta(:)) .* basis_v;
    surface_min = min(surface_min, min(ring, [], 1));
    surface_max = max(surface_max, max(ring, [], 1));
end

interior_min = [double(kgrid.x_vec(pml_size + 1)), ...
                double(kgrid.y_vec(pml_size + 1)), ...
                double(kgrid.z_vec(pml_size + 1))];
interior_max = [double(kgrid.x_vec(end - pml_size)), ...
                double(kgrid.y_vec(end - pml_size)), ...
                double(kgrid.z_vec(end - pml_size))];
if any(surface_min < interior_min) || any(surface_max > interior_max)
    error(['The complete tilted bowl surface does not fit inside the non-PML grid. ' ...
           'Increase padding or reduce opts.source_diameter_mm/source_gap_mm.']);
end
end

function source_signal = makeRampSignal(t, f0, amp, ramp_cycles)
if ramp_cycles > 0
    T_ramp = ramp_cycles / f0;
    ramp = min(t / T_ramp, 1);
else
    ramp = ones(size(t));
end
source_signal = amp * cos(2 * pi * f0 * t) .* ramp;
end

function [Nt, timing] = estimateTimeSteps( ...
        kgrid, sensor_vox, array_mask, cmin, opts)
[source_x, source_y, source_z] = ind2sub(size(array_mask), find(array_mask));
if isempty(source_x)
    error('The distributed source mask is empty.');
end

source_min_m = [min(double(kgrid.x_vec(source_x))), ...
                min(double(kgrid.y_vec(source_y))), ...
                min(double(kgrid.z_vec(source_z)))];
source_max_m = [max(double(kgrid.x_vec(source_x))), ...
                max(double(kgrid.y_vec(source_y))), ...
                max(double(kgrid.z_vec(source_z)))];
sensor_min_m = [min(double(kgrid.x_vec(sensor_vox(:,1)))), ...
                min(double(kgrid.y_vec(sensor_vox(:,2)))), ...
                min(double(kgrid.z_vec(sensor_vox(:,3))))];
sensor_max_m = [max(double(kgrid.x_vec(sensor_vox(:,1)))), ...
                max(double(kgrid.y_vec(sensor_vox(:,2)))), ...
                max(double(kgrid.z_vec(sensor_vox(:,3))))];

% The bounding-box diagonal is a conservative upper bound on the distance
% from any BLI source-support voxel to any recorded sensor voxel.
axis_distance_m = max(abs(sensor_min_m - source_max_m), ...
    abs(sensor_max_m - source_min_m));
max_direct_distance_m = norm(axis_distance_m);
direct_arrival_upper_bound_s = max_direct_distance_m / cmin;

configured_cycles = opts.ramp_cycles + opts.steady_cycles + opts.extra_cycles;
steady_check_cycles = opts.steady_state_settle_cycles + ...
    opts.steady_state_check_windows * opts.nCycles_win;
post_arrival_cycles = max(configured_cycles, steady_check_cycles);
requested_end_time_s = direct_arrival_upper_bound_s + post_arrival_cycles / opts.f0;
Nt = ceil(requested_end_time_s / kgrid.dt) + 1;

timing = struct();
timing.method = 'source-support bounding-box upper bound';
timing.max_direct_distance_m = max_direct_distance_m;
timing.direct_arrival_upper_bound_s = direct_arrival_upper_bound_s;
timing.post_arrival_cycles = post_arrival_cycles;
timing.steady_state_settle_cycles = opts.steady_state_settle_cycles;
timing.steady_state_check_windows = opts.steady_state_check_windows;
timing.window_cycles = opts.nCycles_win;
timing.requested_end_time_s = requested_end_time_s;
timing.Nt = Nt;
fprintf(['Conservative time record: direct-arrival upper bound %.3f us, ' ...
         'then %.1f cycles, Nt = %d.\n'], ...
    1e6 * direct_arrival_upper_bound_s, post_arrival_cycles, Nt);
end

function [P_3d, I_3d, p_vol_t_save, steady_state] = extractSteadyStateField( ...
    sensor_data, locS, local_size, t, dt, f0, speed_map, density_map, ...
    sensor_vox, array_mask, opts)
if isstruct(sensor_data)
    p_raw = sensor_data.p;
else
    p_raw = sensor_data;
end
if isa(p_raw, 'gpuArray')
    p_raw = gather(p_raw);
end

p_use = double(p_raw(locS, :));
if opts.save_time_series
    p_vol_t_save = single(p_use);
else
    p_vol_t_save = [];
end
Nt = numel(t);
t = double(t(:));

omega = 2 * pi * f0;
Nwin = max(16, round(opts.nCycles_win / (f0 * dt)));
w = localHann(Nwin);
w_sum = sum(w);
window_count = round(opts.steady_state_check_windows);
required_samples = window_count * Nwin;
if Nt < required_samples
    error('The time record is too short for %d complete %.3g-cycle windows.', ...
        window_count, opts.nCycles_win);
end

window_spans = zeros(window_count, 2);
P_windows = complex(zeros(size(p_use, 1), window_count));
for window_idx = 1:window_count
    window_end = Nt - (window_count - window_idx) * Nwin;
    window_start = window_end - Nwin + 1;
    span = window_start:window_end;
    window_spans(window_idx, :) = [window_start window_end];
    P_windows(:, window_idx) = projectFrequencyWindow( ...
        p_use, t, span, omega, w, w_sum, opts.postprocess_chunk_points);
end

sensor_linear = sub2ind(size(array_mask), sensor_vox(:,1), ...
    sensor_vox(:,2), sensor_vox(:,3));
non_source = ~array_mask(sensor_linear);
if ~any(non_source)
    error('The sensor plane contains no non-source points for convergence checking.');
end
non_source_windows = P_windows(non_source, :);
if any(~isfinite(non_source_windows(:)))
    error('The steady-state windows contain non-finite non-source pressure values.');
end

window_peak = max(abs(P_windows), [], 2);
field_peak = max(window_peak(non_source));
minimum_field_peak = max(eps, 1e-12 * opts.source_amp);
if ~isfinite(field_peak) || field_peak <= minimum_field_peak
    error(['The steady-state windows produced an empty non-source pressure ' ...
           'field (peak %.4g Pa). No visualization was saved.'], field_peak);
end
active_threshold = field_peak * 10^(opts.steady_state_check_floor_db / 20);
active = non_source & window_peak >= active_threshold;
if ~any(active)
    error('No non-source sensor points passed the steady-state check floor.');
end

complex_errors = zeros(window_count - 1, 1);
amplitude_errors = zeros(window_count - 1, 1);
for transition_idx = 1:(window_count - 1)
    previous_field = P_windows(active, transition_idx);
    current_field = P_windows(active, transition_idx + 1);
    complex_errors(transition_idx) = norm(current_field - previous_field) / ...
        max(norm(current_field), eps);
    amplitude_errors(transition_idx) = ...
        norm(abs(current_field) - abs(previous_field)) / ...
        max(norm(abs(current_field)), eps);
    fprintf(['Steady-state windows %d -> %d: complex change %.3f%%, ' ...
             'amplitude change %.3f%% (limit %.3f%%).\n'], ...
        transition_idx, transition_idx + 1, ...
        100 * complex_errors(transition_idx), ...
        100 * amplitude_errors(transition_idx), ...
        100 * opts.steady_state_tolerance);
end

max_complex_error = max(complex_errors);
max_amplitude_error = max(amplitude_errors);
converged = all(isfinite(complex_errors)) && all(isfinite(amplitude_errors)) && ...
    max_complex_error <= opts.steady_state_tolerance && ...
    max_amplitude_error <= opts.steady_state_tolerance;
if ~converged
    error(['The final %d steady-state windows are not converged: maximum ' ...
           'complex change %.3f%% and amplitude change %.3f%%; limit %.3f%%. ' ...
           'Increase opts.steady_state_settle_cycles, check PML convergence, ' ...
           'or add physically calibrated attenuation.'], window_count, ...
        100 * max_complex_error, 100 * max_amplitude_error, ...
        100 * opts.steady_state_tolerance);
end

P_vec = P_windows(:, end);
steady_state = struct();
steady_state.converged = converged;
steady_state.relative_complex_error = max_complex_error;
steady_state.relative_amplitude_error = max_amplitude_error;
steady_state.complex_errors = complex_errors;
steady_state.amplitude_errors = amplitude_errors;
steady_state.tolerance = opts.steady_state_tolerance;
steady_state.window_spans = window_spans;
steady_state.window_times_s = [t(window_spans(:,1)), t(window_spans(:,2))];
steady_state.analysis_span = window_spans(end, :);
steady_state.analysis_time_s = t(window_spans(end, :)).';
steady_state.window_cycles = opts.nCycles_win;
steady_state.actual_window_cycles = (Nwin - 1) * dt * f0;
steady_state.check_windows = window_count;
steady_state.check_floor_db = opts.steady_state_check_floor_db;
steady_state.checked_point_count = nnz(active);
steady_state.non_source_point_count = nnz(non_source);

P_3d = reshape(P_vec, local_size);

lin_global = sub2ind(size(speed_map), sensor_vox(:,1), sensor_vox(:,2), sensor_vox(:,3));
c_local = reshape(double(speed_map(lin_global)), local_size);
rho_local = reshape(double(density_map(lin_global)), local_size);
I_3d = (abs(P_3d).^2) ./ max(2 .* rho_local .* c_local, eps);
end

function P_vec = projectFrequencyWindow( ...
        p_use, t, span, omega, w, w_sum, chunk_points)
reference = w .* exp(-1i * omega * t(span));
N_points = size(p_use, 1);
P_vec = complex(zeros(N_points, 1));
for first_point = 1:chunk_points:N_points
    last_point = min(first_point + chunk_points - 1, N_points);
    rows = first_point:last_point;
    segment = p_use(rows, span);
    segment = segment - mean(segment, 2);
    P_vec(rows) = (2 / max(w_sum, eps)) .* (segment * reference);
end
end

function [peak_vox, peak_pressure, target_pressure, offset_mm] = ...
        summarizePressureField(P_3d, array_mask, sensor_vox, target_vox, voxelSize)
field_abs = abs(P_3d(:));
sensor_linear = sub2ind(size(array_mask), sensor_vox(:,1), ...
    sensor_vox(:,2), sensor_vox(:,3));
field_abs(array_mask(sensor_linear)) = -Inf;
[peak_pressure, peak_linear] = max(field_abs);
if ~isfinite(peak_pressure)
    error('No finite non-source pressure values were produced on the sensor plane.');
end
peak_vox = sensor_vox(peak_linear, :);

target_linear = find(all(bsxfun(@eq, sensor_vox, target_vox), 2), 1);
if isempty(target_linear)
    error('The target voxel is not included in the sensor plane.');
end
target_pressure = abs(P_3d(target_linear));
offset_mm = (double(peak_vox) - double(target_vox)) .* voxelSize;
end

function savePressureSlice(P_3d, skull_mask, array_mask, target_vox, ...
    source_info, local_axes_mm, sensor_vox, voxelSize, opts)
png_file = opts.png_file;
if isempty(png_file)
    return;
end

slice_y = round(median(sensor_vox(:,2)));
ix = unique(sensor_vox(:,1)).';
iz = unique(sensor_vox(:,3)).';
x_mm = double(local_axes_mm{1});
z_mm = double(local_axes_mm{3});
field_xz = squeeze(double(P_3d(:,1,:)));
field_xz(~isfinite(field_xz)) = 0;

skull_slice = squeeze(skull_mask(ix, slice_y, iz));
source_slice = squeeze(array_mask(ix, slice_y, iz));

if opts.visualize_forward_only && isfield(source_info, 'bowl_vox_approx')
    bowl_x_mm = (source_info.bowl_vox_continuous(1) - 1) * voxelSize(1);
    bowl_z_mm = (source_info.bowl_vox_continuous(3) - 1) * voxelSize(3);
    direction_xz = source_info.incident_direction([1 3]);
    [Xmm, Zmm] = ndgrid(x_mm, z_mm);
    forward_distance_mm = (Xmm - bowl_x_mm) .* direction_xz(1) + ...
        (Zmm - bowl_z_mm) .* direction_xz(2);
    field_xz(forward_distance_mm < -opts.visualize_z_margin_mm) = 0;
end

normalization_values = field_xz(~source_slice & field_xz > 0);
if isempty(normalization_values)
    field_max = max(field_xz(:));
else
    field_max = max(normalization_values);
end
if field_max > 0
    field_db = 20 * log10(field_xz / field_max + eps);
else
    field_db = zeros(size(field_xz));
end
field_db = max(field_db, opts.visualize_db_floor);

fig = figure('Visible', 'off', 'Color', 'w');
ax = axes(fig);
imagesc(ax, x_mm, z_mm, field_db.');
set(ax, 'YDir', 'normal');
axis(ax, 'image');
hold(ax, 'on');
if exist('turbo', 'file') || exist('turbo', 'builtin')
    colormap(ax, turbo(256));
else
    colormap(ax, parula(256));
end
caxis(ax, [opts.visualize_db_floor 0]);
cb = colorbar(ax);
cb.Label.String = '|P_{f0}| (dB, normalized)';

if any(skull_slice(:))
    contour(ax, x_mm, z_mm, double(skull_slice.'), [0.5 0.5], ...
        'Color', [0.85 0.85 1.0], 'LineWidth', 0.8);
end
if isfield(source_info, 'bowl_vox_approx') && ...
        isfield(source_info, 'roc_m') && isfield(source_info, 'diameter_m')
    bowl_x_mm = (source_info.bowl_vox_continuous(1) - 1) * voxelSize(1);
    bowl_z_mm = (source_info.bowl_vox_continuous(3) - 1) * voxelSize(3);
    roc_mm = source_info.roc_m * 1e3;
    half_diameter_mm = source_info.diameter_m * 1e3 / 2;
    lateral_mm = linspace(-half_diameter_mm, half_diameter_mm, 200);
    sag_mm = roc_mm - sqrt(max(roc_mm^2 - lateral_mm.^2, 0));
    if isfield(source_info, 'incident_direction')
        direction_xz = source_info.incident_direction([1 3]);
    else
        direction_xz = [0 -1];
    end
    lateral_xz = [direction_xz(2), -direction_xz(1)];
    bowl_curve_x_mm = bowl_x_mm + ...
        lateral_mm * lateral_xz(1) + sag_mm * direction_xz(1);
    bowl_curve_z_mm = bowl_z_mm + ...
        lateral_mm * lateral_xz(2) + sag_mm * direction_xz(2);
    plot(ax, bowl_curve_x_mm, bowl_curve_z_mm, ...
        'k-', 'LineWidth', 1.8);
end
if field_max > 0 && min(field_db(:)) <= -6 && max(field_db(:)) >= -6
    contour(ax, x_mm, z_mm, field_db.', [-6 -6], ...
        'Color', [1 1 1], 'LineWidth', 1.4);
end

target_x_mm = (target_vox(1) - 1) * voxelSize(1);
target_z_mm = (target_vox(3) - 1) * voxelSize(3);
plot(ax, target_x_mm, target_z_mm, 'wx', 'MarkerSize', 10, 'LineWidth', 1.6);

if isfield(source_info, 'bowl_vox_approx')
    bowl_x_mm = (source_info.bowl_vox_continuous(1) - 1) * voxelSize(1);
    bowl_z_mm = (source_info.bowl_vox_continuous(3) - 1) * voxelSize(3);
    plot(ax, bowl_x_mm, bowl_z_mm, 'wo', 'MarkerSize', 5, 'LineWidth', 1.2);
end

xlabel(ax, 'x (mm)');
ylabel(ax, 'z (mm)');
title(ax, sprintf('y = %.1f mm', (slice_y - 1) * voxelSize(2)));
set(ax, 'Box', 'on', 'Layer', 'top');

png_file = char(png_file);
png_dir = fileparts(png_file);
if ~isempty(png_dir) && ~exist(png_dir, 'dir')
    mkdir(png_dir);
end
if exist('exportgraphics', 'file')
    exportgraphics(fig, png_file, 'Resolution', 220);
else
    print(fig, png_file, '-dpng', '-r220');
end
close(fig);
fprintf('Saved y-center slice visualization to %s\n', png_file);
end

function info = stripSourceInfo(info)
if isfield(info, 'karray')
    info = rmfield(info, 'karray');
end
end

function idx = nearestIndex(vec, value)
[~, idx] = min(abs(double(vec(:)) - value));
end

function sensor_data = runKWaveSolver(kgrid, medium, source, sensor, input_args)
if exist('kspaceFirstOrder3DG', 'file')
    try
        cuda_args = addKWaveBinaryPath(input_args);
        fprintf('Running kspaceFirstOrder3DG (CUDA binary)...\n');
        sensor_data = kspaceFirstOrder3DG(kgrid, medium, source, sensor, cuda_args{:});
        return;
    catch MEcuda
        warning('kspaceFirstOrder3DG failed (%s). Retrying with kspaceFirstOrder3D.', MEcuda.message);
    end
else
    warning('kspaceFirstOrder3DG not found; using kspaceFirstOrder3D.');
end

try
    fprintf('Running kspaceFirstOrder3D...\n');
    sensor_data = kspaceFirstOrder3D(kgrid, medium, source, sensor, input_args{:});
    return;
catch MEgpu
    warning('kspaceFirstOrder3D GPU run failed (%s). Retrying with CPU single.', MEgpu.message);
end

cpu_args = removeNameValueArgs(input_args, {'DataCast', 'DataRecast'});
cpu_args = [cpu_args, {'DataCast', 'single', 'DataRecast', true}];
sensor_data = kspaceFirstOrder3D(kgrid, medium, source, sensor, cpu_args{:});
end

function argsOut = addKWaveBinaryPath(argsIn)
argsOut = argsIn;
if hasNameValueArg(argsOut, 'BinaryPath')
    return;
end
binaryPath = findKWaveBinaryPath();
if ~isempty(binaryPath)
    argsOut = [argsOut, {'BinaryPath', binaryPath}];
end
end

function tf = hasNameValueArg(argsIn, name)
tf = false;
for ii = 1:numel(argsIn)
    if (ischar(argsIn{ii}) || isstring(argsIn{ii})) && strcmpi(char(argsIn{ii}), name)
        tf = true;
        return;
    end
end
end

function binaryPath = findKWaveBinaryPath()
binaryPath = "";
kwFun = which('kspaceFirstOrder3DG');
if isempty(kwFun)
    return;
end
kwRoot = fileparts(kwFun);
candidates = {fullfile(kwRoot, 'binaries'), fullfile(fileparts(kwRoot), 'binaries')};
for ii = 1:numel(candidates)
    if exist(candidates{ii}, 'dir')
        binaryPath = candidates{ii};
        return;
    end
end
end

function argsOut = removeNameValueArgs(argsIn, names)
argsOut = {};
i = 1;
while i <= numel(argsIn)
    key = argsIn{i};
    if (ischar(key) || isstring(key)) && any(strcmpi(char(key), names)) && i < numel(argsIn)
        i = i + 2;
    else
        argsOut{end+1} = key; %#ok<AGROW>
        i = i + 1;
    end
end
end

function w = localHann(n)
if exist('hann', 'file')
    w = hann(n);
elseif exist('hanning', 'file')
    w = hanning(n);
else
    if n <= 1
        w = ones(n, 1);
    else
        idx = (0:n-1).';
        w = 0.5 - 0.5 * cos(2 * pi * idx / (n - 1));
    end
end
end
