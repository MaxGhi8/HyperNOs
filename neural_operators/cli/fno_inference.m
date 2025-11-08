% FNO_INFERENCE - Load and run inference with exported FNO model
%
% This script demonstrates how to load an FNO model exported from PyTorch
% and perform forward pass inference in MATLAB.
%
% Usage:
%   1. Export your PyTorch model using export_fno_to_mat.py
%   2. Modify the parameters below to match your setup
%   3. Run this script: fno_inference
%
% Author: Generated for HyperNOs project
% Date: 2025-11-07

clear; close all; clc;

%% Configuration
% Path to the exported .mat file with model parameters
model_file = '../tests/FNO/afieti_fno/loss_L2_mode_default/trained_params.mat';

% Load the model parameters
fprintf('Loading FNO model from: %s\n', model_file);
model = load(model_file);

% Display model metadata
fprintf('\n=== Model Architecture ===\n');
fprintf('Model Type: %s\n', model.model_type);
fprintf('Problem Dimension: %d\n', model.problem_dim);
fprintf('Input Dimension: %d\n', model.in_dim);
fprintf('Hidden Dimension (d_v): %d\n', model.d_v);
fprintf('Output Dimension: %d\n', model.out_dim);
fprintf('Number of Layers (L): %d\n', model.L);
fprintf('Fourier Modes: %d\n', model.modes);
fprintf('Activation Function: %s\n', model.fun_act);
fprintf('Architecture Type: %s\n', model.arc);
fprintf('RNN: %d\n', model.RNN);
fprintf('Padding: %d\n', model.padding);

%% Example of data preparation
% Check if test batch is included in the exported model
if isfield(model, 'has_test_batch') && model.has_test_batch
    fprintf('\n=== Using exported test batch ===\n');
    x_input = model.test_X_batch;
    x_input(1, 1, 1)
    y_target = model.test_y_batch;
    y_pred_pytorch = model.test_y_pred_pytorch;

    fprintf('Test batch loaded from exported model\n');
    if model.problem_dim == 1
        fprintf('Input shape: [%d, %d, %d]\n', size(x_input, 1), size(x_input, 2), size(x_input, 3));
    elseif model.problem_dim == 2
        fprintf('Input shape: [%d, %d, %d, %d]\n', size(x_input, 1), size(x_input, 2), size(x_input, 3), size(x_input, 4));
    end
else
    % Create dummy input if no test batch available
    fprintf('\n=== Creating dummy input (no test batch in model) ===\n');
    n_samples = 12;
    n_points = 100;
    in_dim = model.in_dim - model.problem_dim; % Subtract grid dimension

    % Create dummy input
    if model.problem_dim == 1
        x_input = randn(n_samples, n_points, in_dim);
    elseif model.problem_dim == 2
        x_input = randn(n_samples, n_points, n_points, in_dim);
    end
    y_pred_pytorch = [];
end

fprintf('\n=== Running Inference ===\n');
if model.problem_dim == 1
    fprintf('Input shape: [%d, %d, %d]\n', size(x_input, 1), size(x_input, 2), size(x_input, 3));
elseif model.problem_dim == 2
    fprintf('Input shape: [%d, %d, %d, %d]\n', size(x_input, 1), size(x_input, 2), size(x_input, 3), size(x_input, 4));
end

%% Run FNO forward pass
output = fno_forward(x_input, model);

if model.problem_dim == 1
    fprintf('Output shape: [%d, %d, %d]\n', size(output, 1), size(output, 2), size(output, 3));
elseif model.problem_dim == 2
    fprintf('Output shape: [%d, %d, %d, %d]\n', size(output, 1), size(output, 2), size(output, 3), size(output, 4));
end
fprintf('\nInference completed successfully!\n');

% Compare with PyTorch if available
if ~isempty(y_pred_pytorch)
    fprintf('\n=== Comparing MATLAB vs PyTorch ===\n');
    abs_diff = abs(output - y_pred_pytorch);

    fprintf('Absolute difference:\n');
    fprintf('  Max:  %.6e\n', max(abs_diff(:)));
    fprintf('  Mean: %.6e\n', mean(abs_diff(:)));

    % Correlation
    corr_coef = corrcoef(y_pred_pytorch(:), output(:));
    fprintf('\nCorrelation coefficient: %.10f\n', corr_coef(1, 2));

    % Check tolerance
    tol = 1e-5;
    if max(abs_diff(:)) < tol
        fprintf('\n✓ SUCCESS: MATLAB and PyTorch outputs match within tolerance (%.e)!\n', tol);
    else
        fprintf('\n⚠ WARNING: Difference exceeds tolerance (%.e)\n', tol);
    end
end

%% Visualization (optional)
if model.problem_dim == 1
    figure('Name', 'FNO Inference Results');

    if ~isempty(y_pred_pytorch)
        % Compare MATLAB vs PyTorch
        subplot(3, 1, 1);
        plot(squeeze(x_input(1, :, :)));
        title('Input');
        xlabel('Spatial Points');
        ylabel('Value');
        grid on;

        subplot(3, 1, 2);
        hold on;
        plot(squeeze(output(1, :, :)), 'LineWidth', 2, 'DisplayName', 'MATLAB');
        plot(squeeze(y_pred_pytorch(1, :, :)), '--', 'LineWidth', 2, 'DisplayName', 'PyTorch');
        hold off;
        title('FNO Output: MATLAB vs PyTorch');
        xlabel('Spatial Points');
        ylabel('Value');
        legend();
        grid on;

        subplot(3, 1, 3);
        plot(squeeze(abs(output(1, :, :) - y_pred_pytorch(1, :, :))), 'LineWidth', 2);
        title('Absolute Difference |MATLAB - PyTorch|');
        xlabel('Spatial Points');
        ylabel('Difference');
        grid on;
    else
        % Show only MATLAB output
        subplot(2, 1, 1);
        plot(squeeze(x_input(1, :, :)));
        title('Input');
        xlabel('Spatial Points');
        ylabel('Value');
        grid on;

        subplot(2, 1, 2);
        plot(squeeze(output(1, :, :)));
        title('FNO Output (MATLAB)');
        xlabel('Spatial Points');
        ylabel('Value');
        grid on;
    end
elseif model.problem_dim == 2 && ~isempty(y_pred_pytorch)
    % 2D comparison visualization
    figure('Name', 'FNO Inference Results - 2D', 'Position', [100, 100, 1200, 800]);

    sample_idx = 1;
    channel_idx = 1;

    subplot(2, 3, 1);
    imagesc(squeeze(x_input(sample_idx, :, :, channel_idx)));
    colorbar;
    title('Input (Sample 1)');
    axis equal tight;

    subplot(2, 3, 2);
    imagesc(squeeze(output(sample_idx, :, :, channel_idx)));
    colorbar;
    title('MATLAB Output');
    axis equal tight;

    subplot(2, 3, 3);
    imagesc(squeeze(y_pred_pytorch(sample_idx, :, :, channel_idx)));
    colorbar;
    title('PyTorch Output');
    axis equal tight;

    subplot(2, 3, 4);
    imagesc(squeeze(abs(output(sample_idx, :, :, channel_idx) - y_pred_pytorch(sample_idx, :, :, channel_idx))));
    colorbar;
    title('Absolute Difference');
    axis equal tight;

    % Difference histogram
    subplot(2, 3, 5:6);
    histogram(abs(output(:) - y_pred_pytorch(:)), 50);
    xlabel('Absolute Difference');
    ylabel('Frequency');
    title('Distribution of Absolute Differences (All Samples)');
    grid on;
end

%% Normalization functions (Gaussian)
function x_normalized = normalize_input_gaussian(x, model, problem_dim)
    % NORMALIZE_INPUT - Apply UnitGaussianNormalizer encoding
    % x = (x - mean) / (std + eps)

    mean = model.input_normalizer_mean;
    std = model.input_normalizer_std;
    eps = model.input_normalizer_eps;

    if problem_dim == 1
        % x: [n_samples, n_points, in_dim]
        % mean, std: [n_points, in_dim]
        [n_samples, n_points, in_dim] = size(x);
        mean_expanded = repmat(reshape(mean, [1, n_points, 1]), [n_samples, 1, in_dim]);
        std_expanded = repmat(reshape(std, [1, n_points, 1]), [n_samples, 1, in_dim]);
        x_normalized = (x - mean_expanded) ./ (std_expanded + eps);

    elseif problem_dim == 2
        % x: [n_samples, n_x, n_y, in_dim]
        % mean, std: [n_x, n_y]
        [n_samples, n_x, n_y, in_dim] = size(x);
        mean_expanded = repmat(reshape(mean, [1, n_x, n_y, 1]), [n_samples, 1, 1, in_dim]);
        std_expanded = repmat(reshape(std, [1, n_x, n_y, 1]), [n_samples, 1, 1, in_dim]);
        x_normalized = (x - mean_expanded) ./ (std_expanded + eps);

    else
        error('Normalization for 3D not yet implemented');
    end
end

function x_denormalized = denormalize_output_gaussian(x, model, problem_dim)
    % DENORMALIZE_OUTPUT - Apply UnitGaussianNormalizer decoding
    % x = x * (std + eps) + mean

    mean = model.output_normalizer_mean;
    std = model.output_normalizer_std;
    eps = model.output_normalizer_eps;

    if problem_dim == 1
        % x: [n_samples, n_points, out_dim]
        % mean, std: [n_points, out_dim]
        [n_samples, n_points, out_dim] = size(x);
        mean_expanded = repmat(reshape(mean, [1, n_points, 1]), [n_samples, 1, out_dim]);
        std_expanded = repmat(reshape(std, [1, n_points, 1]), [n_samples, 1, out_dim]);
        x_denormalized = x .* (std_expanded + eps) + mean_expanded;

    elseif problem_dim == 2
        % x: [n_samples, n_x, n_y, out_dim]
        % mean, std: [n_x, n_y, out_dim]
        [n_samples, n_x, n_y, out_dim] = size(x);
        mean_expanded = repmat(reshape(mean, [1, n_x, n_y, 1]), [n_samples, 1, 1, out_dim]);
        std_expanded = repmat(reshape(std, [1, n_x, n_y, 1]), [n_samples, 1, 1, out_dim]);
        x_denormalized = x .* (std_expanded + eps) + mean_expanded;

    else
        error('Denormalization for 3D not yet implemented');
    end
end

%% Normalization functions (MinMax)
function x_normalized = normalize_input_minmax(x, model, problem_dim)
    % NORMALIZE_INPUT_MINMAX - Apply minmax normalization
    % x = (x - min) / (max - min)
    % Assumes min_val and max_val are scalars

    min_val = model.input_normalizer_min;
    max_val = model.input_normalizer_max;

    x_normalized = (x - min_val) ./ (max_val - min_val);
end

function x_denormalized = denormalize_output_minmax(x, model, problem_dim)
    % DENORMALIZE_OUTPUT_MINMAX - Apply minmax denormalization
    % x = x * (max - min) + min
    % Assumes min_val and max_val are scalars

    min_val = model.output_normalizer_min;
    max_val = model.output_normalizer_max;

    x_denormalized = x .* (max_val - min_val) + min_val;
end

%% Activation function
function out = apply_activation(x, fun_act)
    switch fun_act
        case 'relu'
            out = max(0, x);
        case 'gelu'
            out = 0.5 * x .* (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x .^ 3)));
        case 'tanh'
            out = tanh(x);
        case 'leaky_relu'
            out = max(0.01 * x, x);
        otherwise
            error('Unknown activation function: %s', fun_act);
    end
end

% Apply activation to tensor
function out = apply_activation_tensor(x, fun_act, problem_dim)
    if problem_dim == 1
        [n_samples, channels, n_points] = size(x);
        x_flat = reshape(x, [n_samples * channels * n_points, 1]);
        out_flat = apply_activation(x_flat, fun_act);
        out = reshape(out_flat, [n_samples, channels, n_points]);

    elseif problem_dim == 2
        [n_samples, channels, n_x, n_y] = size(x);
        x_flat = reshape(x, [n_samples * channels * n_x * n_y, 1]);
        out_flat = apply_activation(x_flat, fun_act);
        out = reshape(out_flat, [n_samples, channels, n_x, n_y]);

    else
        error('Unknown problem dimension: %d', problem_dim);

    end
end

%% Shallow MLP
function out = mlp_forward(problem_dim, x, w1, b1, w2, b2, fun_act)
    if problem_dim == 1
        % reshape
        [n_samples, n_points, ~] = size(x);
        x_flat = reshape(x, [n_samples * n_points, size(x, 3)]);

        % First layer
        h = x_flat * w1' + b1';
        h = apply_activation(h, fun_act);

        % Second layer
        out = h * w2' + b2';

        % Reshape back
        out = reshape(out, [n_samples, n_points, size(out, 2)]);

    elseif problem_dim == 2
        % reshape
        [n_samples, n_x, n_y, ~] = size(x);
        x_flat = reshape(x, [n_samples * n_x * n_y, size(x, 4)]);

        % First layer
        h = x_flat * w1' + b1';
        h = apply_activation(h, fun_act);

        % Second layer
        out = h * w2' + b2';

        % Reshape back
        out = reshape(out, [n_samples, n_x, n_y, size(out, 2)]);
    else
        error('MLP for 3D not yet implemented');
    end
end

% Linear layer - like the conv trick in pytorch
function out = linear_conv(problem_dim, x, w1, b1)
    if problem_dim == 1
        x = permute(x, [1, 3, 2]);

        [n_samples, n_points, ~] = size(x);
        x_flat = reshape(x, [n_samples * n_points, size(x, 3)]);
        out = x_flat * w1' + b1';
        out = reshape(out, [n_samples, n_points, size(out, 2)]);

        out = permute(out, [1, 3, 2]);

    elseif problem_dim == 2
        x = permute(x, [1, 3, 4, 2]);

        [n_samples, n_x, n_y, ~] = size(x);
        x_flat = reshape(x, [n_samples * n_x * n_y, size(x, 4)]);
        out = x_flat * w1' + b1';
        out = reshape(out, [n_samples, n_x, n_y, size(out, 2)]);

        out = permute(out, [1, 4, 2, 3]);

    else
        error('MLP_conv for 3D not yet implemented');
    end

end

% MLP_conv - MLP with convolutional structure
function out = mlp_conv_forward(problem_dim, x, w1, b1, w2, b2, fun_act)
    % x shape is [n_samples, channels, spatial_points*]

    if problem_dim == 1
        x = permute(x, [1, 3, 2]);
        x = mlp_forward(problem_dim, x, w1, b1, w2, b2, fun_act);
        out = permute(x, [1, 3, 2]);

    elseif problem_dim == 2
        x = permute(x, [1, 3, 4, 2]);
        x = mlp_forward(problem_dim, x, w1, b1, w2, b2, fun_act);
        out = permute(x, [1, 4, 2, 3]);

    else
        error('MLP_conv for 3D not yet implemented');
    end

end

%% FNO Forward Pass
function output = fno_forward(x, model)
    problem_dim = model.problem_dim;
    L = model.L;
    padding = model.padding;
    fun_act = model.fun_act;

    % Input normalization (if available)
    % if isfield(model, 'has_input_normalizer_gaussian') && model.has_input_normalizer_gaussian
    %     x = normalize_input_gaussian(x, model, problem_dim);
    % elseif isfield(model, 'has_input_normalizer_minmax') && model.has_input_normalizer_minmax
    %     x = normalize_input_minmax(x, model, problem_dim);
    % end

    % Add grid coordinates to input
    x = add_grid(x, problem_dim);

    % Lifting operator P (MLP)
    x = mlp_forward(problem_dim, x, model.p_mlp1_weight, model.p_mlp1_bias, ...
        model.p_mlp2_weight, model.p_mlp2_bias, fun_act);

    % Reshape for convolution: [n_samples, spatial_points*, d_v] -> [n_samples, d_v, spatial_points*]
    if problem_dim == 1
        x = permute(x, [1, 3, 2]);
    elseif problem_dim == 2
        x = permute(x, [1, 4, 2, 3]);
    end

    % Apply padding
    if padding > 0
        x = pad_tensor(x, padding, problem_dim);
    end

    % Fourier Layers
    for i = 0:(L - 1)
        x = fourier_layer(x, model, i);
    end

    x(1, 1:11, 1, 1)

    % Remove padding
    if padding > 0
        x = unpad_tensor(x, padding, problem_dim);
    end

    % Reshape back: [n_samples, d_v, spatial_points] -> [n_samples, spatial_points, d_v]
    if problem_dim == 1
        x = permute(x, [1, 3, 2]);
    elseif problem_dim == 2
        x = permute(x, [1, 3, 4, 2]);
    end

    % Projection operator Q (MLP)
    output = mlp_forward(problem_dim, x, model.q_mlp1_weight, model.q_mlp1_bias, ...
        model.q_mlp2_weight, model.q_mlp2_bias, fun_act);

    % Output denormalization (if available)
    if isfield(model, 'has_output_normalizer_gaussian') && model.has_output_normalizer_gaussian
        output = denormalize_output_gaussian(output, model, problem_dim);
    elseif isfield(model, 'has_output_normalizer_minmax') && model.has_output_normalizer_minmax
        output = denormalize_output_minmax(output, model, problem_dim);
    end

end

%% UTILS for FNO forward pass: add grid function
function x = add_grid(x, problem_dim)
    if problem_dim == 1
        [n_samples, n_points, ~] = size(x);
        grid = linspace(0, 1, n_points)';
        grid = repmat(reshape(grid, [1, n_points, 1]), [n_samples, 1, 1]);
        x = cat(3, grid, x);

    elseif problem_dim == 2
        [n_samples, size_x, size_y, ~] = size(x);

        % Grid for x dimension: shape [1, size_x, 1, 1] repeated to [n_samples, size_x, size_y, 1]
        gridx = linspace(0, 1, size_x);
        gridx = reshape(gridx, [1, size_x, 1, 1]);
        gridx = repmat(gridx, [n_samples, 1, size_y, 1]);

        % Grid for y dimension: shape [1, 1, size_y, 1] repeated to [n_samples, size_x, size_y, 1]
        gridy = linspace(0, 1, size_y);
        gridy = reshape(gridy, [1, 1, size_y, 1]);
        gridy = repmat(gridy, [n_samples, size_x, 1, 1]);

        % Concatenate grids with input: [n_samples, size_x, size_y, 2+in_dim]
        x = cat(4, gridx, gridy, x);

    elseif problem_dim == 3
        warning('3D grid addition needs to be customized for your specific case');
    end
end

%% UTILS for FNO forward pass: padding and unpadding functions
function x_padded = pad_tensor(x, padding, problem_dim)
    if problem_dim == 1
        [n_samples, channels, n_points] = size(x);
        x_padded = zeros(n_samples, channels, n_points + 2 * padding);
        x_padded(:, :, padding + 1:n_points + padding) = x;
    elseif problem_dim == 2
        [n_samples, channels, n_x, n_y] = size(x);
        x_padded = zeros(n_samples, channels, n_x + 2 * padding, n_y + 2 * padding);
        x_padded(:, :, padding + 1:n_x + padding, padding + 1:n_y + padding) = x;
    else
        error('Padding for 3D not yet implemented');
    end
end

function x_unpadded = unpad_tensor(x, padding, problem_dim)
    if problem_dim == 1
        [~, ~, n_points] = size(x);
        x_unpadded = x(:, :, padding + 1:n_points + padding);
    else
        [~, ~, n_x, n_y] = size(x);
        x_unpadded = x(:, :, padding + 1:n_x + padding, padding + 1:n_y + padding);
    end
end

%% Fourier Layer
function x = fourier_layer(x, model, layer_idx)

    problem_dim = model.problem_dim;
    modes = model.modes;
    arc = model.arc;
    fun_act = model.fun_act;
    RNN = model.RNN;
    L = model.L;

    % Computes Fourier layer
    if RNN
        weights_key = 'integrals_weights';
    else
        weights_key = sprintf('integrals_%d_weights', layer_idx);
    end

    if problem_dim == 1
        if isfield(model, [weights_key '_real'])
            fourier_weights = model.([weights_key '_real']) + 1i * model.([weights_key '_imag']);
        else
            error('Fourier weights not found: %s', weights_key);
        end
        x_fourier = fourier_transform_1d(x, fourier_weights, modes);

    elseif problem_dim == 2
        weights1_key = sprintf('integrals_%d_weights1', layer_idx);
        weights2_key = sprintf('integrals_%d_weights2', layer_idx);

        fourier_weights1 = model.([weights1_key '_real']) + 1i * model.([weights1_key '_imag']);
        fourier_weights2 = model.([weights2_key '_real']) + 1i * model.([weights2_key '_imag']);

        x_fourier = fourier_transform_2d(x, fourier_weights1, fourier_weights2, modes);

    else
        error('3D Fourier layer not yet implemented');
    end

    % Computes skip connection
    if RNN
        ws_key = 'ws';
    else
        ws_key = sprintf('ws_%d', layer_idx);
    end

    ws_weight = model.([ws_key '_weight']);
    ws_bias = model.([ws_key '_bias']);

    x_skip = linear_conv(problem_dim, x, ws_weight, ws_bias);

    % Combine based on architecture
    switch arc
        case 'Classic'
            x = x_fourier + x_skip;
            if layer_idx < L - 1
                x = apply_activation_tensor(x, fun_act, problem_dim);
            end

        case 'Zongyi'
            % Get MLP weights
            if RNN
                mlp_key = 'mlps';
            else
                mlp_key = sprintf('mlps_%d', layer_idx);
            end

            mlp1_weight = model.([mlp_key '_mlp1_weight']);
            mlp1_bias = model.([mlp_key '_mlp1_bias']);
            mlp2_weight = model.([mlp_key '_mlp2_weight']);
            mlp2_bias = model.([mlp_key '_mlp2_bias']);

            % Apply Zongyi variant
            x1 = mlp_conv_forward(problem_dim, x_fourier, mlp1_weight, mlp1_bias, ...
                mlp2_weight, mlp2_bias, fun_act);

            if layer_idx == 0
                x1(1, 1:11, 1, 1)
                x_skip(1, 1:11, 1, 1)
            end

            x = x1 + x_skip;
            if layer_idx < L - 1
                x = apply_activation_tensor(x, fun_act, problem_dim);
            end

        case 'Residual'
            warning('Residual architecture not implemented');

        case 'Tran'
            warning('Tran architecture not implemented');

        otherwise
            error('Unknown architecture: %s', arc);
    end
end

%% Fourier Transform function
function out = fourier_transform_1d(x, weights, modes)
    [n_samples, in_channels, n_points] = size(x);
    out_channels = size(weights, 2);

    % FFT
    x_fft = fft(x, [], 3);

    % Initialize output in Fourier space
    out_fft = zeros(n_samples, out_channels, n_points);

    % Apply Fourier weights
    for sample = 1:n_samples
        for out_ch = 1:out_channels
            for mode = 1:modes
                for in_ch = 1:in_channels
                    out_fft(sample, out_ch, mode) = out_fft(sample, out_ch, mode) + ...
                        x_fft(sample, in_ch, mode) * weights(in_ch, out_ch, mode);
                end
            end
        end
    end

    % Inverse FFT
    out = real(ifft(out_fft, [], 3));
end

% 2D Fourier Layer function
function out = fourier_transform_2d(x, weights1, weights2, modes)
    [n_samples, in_channels, n_x, n_y] = size(x);
    out_channels = size(weights1, 2);

    % ========================================================================
    % 2D FFT
    x_fft = fft(fft(x, [], 3), [], 4);

    % Impose Hermitian symmetry
    n_y_rfft = floor(n_y / 2) + 1;
    x_fft = x_fft(:, :, :, 1:n_y_rfft);

    % Initialize output in Fourier space
    out_fft = zeros(n_samples, out_channels, n_x, n_y_rfft);

    % ========================================================================
    % Top-left corner: Apply weights1
    % Extract relevant portion: [n_samples, in_channels, modes, modes]
    x_fft_corner1 = x_fft(:, :, 1:modes, 1:modes);

    % Tensor contraction using reshape and matrix multiply
    % [n_samples, in_channels, modes, modes] -> [n_samples, modes, modes, in_channels]
    x_perm1 = permute(x_fft_corner1, [1, 3, 4, 2]);
    % [in_channels, out_channels, modes, modes] -> [modes, modes, in_channels, out_channels]
    w1_perm = permute(weights1, [3, 4, 1, 2]);

    % Reshape for matrix multiplication: [n_samples * modes * modes, in_channels]
    x_flat1 = reshape(x_perm1, [n_samples * modes * modes, in_channels]);
    % [modes * modes, in_channels, out_channels]
    w1_flat = reshape(w1_perm, [modes * modes, in_channels, out_channels]);

    out_flat1 = zeros(n_samples * modes * modes, out_channels);

    % Vectorized: for each spatial frequency, multiply all samples at once
    for idx = 1:(modes * modes)
        sample_start = (idx - 1) * n_samples + 1;
        sample_end = idx * n_samples;
        % [n_samples, in_channels] @ [in_channels, out_channels] = [n_samples, out_channels]
        out_flat1(sample_start:sample_end, :) = x_flat1(sample_start:sample_end, :) * squeeze(w1_flat(idx, :, :));
    end

    % Reshape back: [n_samples, modes, modes, out_channels] -> [n_samples, out_channels, modes, modes]
    out_corner1 = reshape(out_flat1, [n_samples, modes, modes, out_channels]);
    out_fft(:, :, 1:modes, 1:modes) = permute(out_corner1, [1, 4, 2, 3]);

    % ========================================================================
    % Bottom-left corner: Apply weights2 (high frequencies in x, low in y)
    x_fft_corner2 = x_fft(:, :, (n_x - modes + 1):n_x, 1:modes);

    x_perm2 = permute(x_fft_corner2, [1, 3, 4, 2]);
    w2_perm = permute(weights2, [3, 4, 1, 2]);

    x_flat2 = reshape(x_perm2, [n_samples * modes * modes, in_channels]);
    w2_flat = reshape(w2_perm, [modes * modes, in_channels, out_channels]);

    out_flat2 = zeros(n_samples * modes * modes, out_channels);

    for idx = 1:(modes * modes)
        sample_start = (idx - 1) * n_samples + 1;
        sample_end = idx * n_samples;
        out_flat2(sample_start:sample_end, :) = x_flat2(sample_start:sample_end, :) * squeeze(w2_flat(idx, :, :));
    end

    out_corner2 = reshape(out_flat2, [n_samples, modes, modes, out_channels]);
    out_fft(:, :, (n_x - modes + 1):n_x, 1:modes) = permute(out_corner2, [1, 4, 2, 3]);

    % ========================================================================
    % Inverse 2D FFT
    out_fft_full = zeros(n_samples, out_channels, n_x, n_y);
    out_fft_full(:, :, :, 1:n_y_rfft) = out_fft;

    % Impose Hermitian symmetry for the output
    if n_y_rfft < n_y
        mirror_indices_from = 2:(n_y_rfft - 1);
        mirror_indices_to = (n_y - mirror_indices_from + 2);
        out_fft_full(:, :, :, mirror_indices_to) = conj(out_fft(:, :, :, mirror_indices_from));
    end

    % Inverse FFT on both dimensions
    out = real(ifft(ifft(out_fft_full, [], 4), [], 3));
end
