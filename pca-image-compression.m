clear;
close all;

data_path = 'unpadded';

X_list = [];   % will hold vectorized images as columns
labels = [];   % store subject IDs

for s = 1:13
    
    subj_prefix = sprintf('subject%02d', s);
    pattern = fullfile(data_path, [subj_prefix '*.pgm']);
    files = dir(pattern);

    for k = 1:numel(files)
        % path to k-th file
        fname = fullfile(files(k).folder, files(k).name);

        img = imread(fname);
        img = double(img); % convert to double for PCA

        img_vec = img(:); % vectorize image into a column vector

        % add vector as a column to X_list
        if isempty(X_list)
            X_list = img_vec;
        else
            X_list(:, end+1) = img_vec;
        end
        labels(end+1,1) = s;
    end
end

X = X_list; % final data matrix X (M × N)

fprintf('\nFinal: %d images loaded.\n', size(X,2));
fprintf('Matrix X size: %d × %d (pixels × images)\n', size(X,1), size(X,2));

mu = mean(X, 2);
Xc = X - mu;
[U, S, V] = svd(Xc, 'econ');

sing_vals = diag(S);
sigma2 = sing_vals.^2;
total_energy = sum(sigma2);
cum_energy = cumsum(sigma2);

percent_error = (total_energy - cum_energy) ./ total_energy;

% plot singular values
figure;
plot(sing_vals, 'o-');
xlabel('Index i');
ylabel('\sigma_i');
title('Singular values of X');

% plot percent error vs k
figure;
plot(percent_error, 'o-');
xlabel('k (number of principal components)');
ylabel('Percent error');
title('Reconstruction error vs. number of components');
grid on;

% thresholds
th30 = 0.30;
th20 = 0.20;
th10 = 0.10;
th05 = 0.05;

k30 = find(percent_error <= th30, 1, 'first');
k20 = find(percent_error <= th20, 1, 'first');
k10 = find(percent_error <= th10, 1, 'first');
k05 = find(percent_error <= th05, 1, 'first');

fprintf('k for <=30%% error: %d\n', k30);
fprintf('k for <=20%% error: %d\n', k20);
fprintf('k for <=10%% error: %d\n', k10);
fprintf('k for <=5%%  error: %d\n', k05);

[M, N] = size(X);   % M pixels, N images

% 10% error already computed in k10, 90% data kept
storage_factor = M / k10;

fprintf('Original per image: %d numbers\n', M);
fprintf('Compressed per image (90%% data): %d numbers\n', k10);
fprintf('Storage reduced by a factor of ~%.2f\n', storage_factor);

ds = [20 50 70 100];

fprintf('\n Dataset percent-error (Frobenius) \n');
for d = ds
    % Frobenius percent error using singular values only: (total - energy of first d) / total
    frob_percent_error = (total_energy - cum_energy(d)) / total_energy;

    fprintf('d = %3d: percent-error = %.4f (%.2f%%)\n', ...
        d, frob_percent_error, 100*frob_percent_error);
end

fprintf('\n Average per-image relative error \n');
for d = ds
    % rank-d reconstruction
    Ud = U(:,1:d);
    Sd = S(1:d,1:d);
    Vd = V(:,1:d);
    Xd = Ud * Sd * Vd';

    rel_err = zeros(N,1);
    for j = 1:N
        xj    = X(:,j);
        xjhat = Xd(:,j);
        rel_err(j) = norm(xj - xjhat)^2 / norm(xj)^2;
    end

    avg_rel_err = mean(rel_err);

    fprintf('d = %3d: avg per-image relative error = %.4f (%.2f%%)\n', ...
        d, avg_rel_err, 100*avg_rel_err);
end

data_path = 'unpadded';
sample_files = dir(fullfile(data_path, 'subject01*.pgm'));
sample_img = imread(fullfile(sample_files(1).folder, sample_files(1).name));
[H, W] = size(sample_img); 

% training image index to visualize
idx = 1; 

% Original image
orig_img_vec = X(:, idx);
orig_img = reshape(orig_img_vec, H, W);

% reconstructions for each d
recon_imgs = cell(length(ds),1);

for t = 1:length(ds)
    d = ds(t);
    Ud = U(:,1:d);
    Sd = S(1:d,1:d);
    Vd = V(:,1:d);

    Xd = Ud * Sd * Vd';   % reconstructed full data matrix

    recon_vec = Xd(:, idx);
    recon_img = reshape(recon_vec, H, W);
    recon_imgs{t} = recon_img;
end

figure;
subplot(1, 1+length(ds), 1);
imagesc(orig_img); colormap gray; axis image off;
title('Original');

for t = 1:length(ds)
    subplot(1, 1+length(ds), t+1);
    imagesc(recon_imgs{t}); colormap gray; axis image off;
    title(sprintf('d = %d', ds(t)));
end

Xtrain = X;

mu = mean(Xtrain, 2); % Mean face (M x 1)

Xc_train = Xtrain - mu; % centered training data

[U, S, V] = svd(Xc_train, 'econ'); % PCA via SVD

Xtest_list = [];
test_labels = [];

for s = 14:15
    subj_prefix = sprintf('subject%02d', s);
    pattern = fullfile(data_path, [subj_prefix '*.pgm']);
    files = dir(pattern);

    for k = 1:numel(files)
        fname = fullfile(files(k).folder, files(k).name);
        img = imread(fname);
        img_vec = double(img(:));

        if isempty(Xtest_list)
            Xtest_list = img_vec;
        else
            Xtest_list(:, end+1) = img_vec;
        end

        test_labels(end+1,1) = s;
    end
end

Xtest = Xtest_list; % M x N_test
fprintf('Test set size: %d x %d (pixels × images)\n', size(Xtest,1), size(Xtest,2));

d = 50;
U50 = U(:,1:d);

Xtest_c = Xtest - mu; % center test data using training mean

% feature values for test images: d x N_test
Ztest = U50' * Xtest_c; % 50-dimensional PCA features

% reconstruct test images from 50-dimensional features
Xtest_hat = mu + U50 * Ztest; % M x N_test

% Frobenius reconstruction error
num = norm(Xtest - Xtest_hat, 'fro')^2;
den = norm(Xtest, 'fro')^2;

test_percent_error_50 = num / den;

fprintf('\nTest set approximation error with d = 50: %.4f (%.2f%%)\n', ...
        test_percent_error_50, 100*test_percent_error_50);

% rotate subject 15 image 25 degrees
orig_fname = fullfile('unpadded', 'subject15.normal.pgm');  
img = imread(orig_fname);
img = double(img);

rot_img = imrotate(img, 25, 'bilinear', 'crop');
rot_img_uint8 = uint8(rot_img);
imwrite(rot_img_uint8, 'subject15rotated.jpeg');

rot_img = imread('subject15rotated.jpeg');

% convert to grayscale 
if ndims(rot_img) == 3
    rot_img = rgb2gray(rot_img);
end
rot_img = double(rot_img);

% make sure rotated image matches training size
rot_img_resized = imresize(rot_img, [H, W]);

x_rot = rot_img_resized(:); % M x 1 
x_rot_c = x_rot - mu; % use training mean to center

% 50d PCA features for rotated image
z_rot = U50' * x_rot_c;   % 50 x 1
x_rot_hat = mu + U50 * z_rot;   % M x 1
num_rot = norm(x_rot - x_rot_hat)^2;
den_rot = norm(x_rot)^2;
rot_percent_error_50 = num_rot / den_rot;

fprintf('Rotated image approximation error with d = 50: %.4f (%.2f%%)\n', ...
        rot_percent_error_50, 100*rot_percent_error_50);

orig_rot_img = reshape(x_rot, H, W);
recon_rot_img = reshape(x_rot_hat, H, W);

figure;
subplot(1,2,1);
imagesc(orig_rot_img); colormap gray; axis image off;
title('Rotated original');

subplot(1,2,2);
imagesc(recon_rot_img); colormap gray; axis image off;
title('Rotated reconstructed (d = 50)');