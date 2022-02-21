%%
clc
clearvars
close all

rng(0)
data_cp = dlmread('./Datasets/concrete_cp.data');

f_type = zeros(1,8); % 0:ordinal 1:nominal

inputs_cp = data_cp(:,1:end-1);
targets_cp = data_cp(:,end);
[n_samples, N] = size(inputs_cp);

p_test = 0.2;

% Give training and validation only
n_sim  = 5;
rmse_test = zeros(1, n_sim); y_test = cell(1, n_sim);
rmse_train = zeros(1, n_sim); y_train = cell(1, n_sim);

md = cell(1, n_sim);
indices = cell(n_sim, 1);
for n_s = 1:n_sim
    indices{n_s} = crossvalind('LeaveMOut', n_samples, n_samples*p_test);
end

for n_s = 1:n_sim
    tr_ind = find(indices{n_s}==1);
    te_ind = find(indices{n_s}==0);
    s_tr = length(tr_ind);
    s_te = length(te_ind);

    % Set parameters
    opts = struct('f_type', f_type, 'I', 25, ...
        'F', [5 10 15] , 'mu', [1e-6 1e-5], 'mu_smooth', [1e-2 0.1 1 10], ...
        'max_itr', 35, 'tol', 1e-3, 'b', 1);

    opts.inputs_tr = inputs_cp(tr_ind, :);
    opts.targets_tr = targets_cp(tr_ind,:);

    opts.inputs_te = inputs_cp(te_ind, :);
    opts.targets_te = targets_cp(te_ind,:);

    md{n_s} = CPD_regression(opts);
    rmse_test(1, n_s) = md{n_s}.test_rmse;
    rmse_test(1, n_s)
end

fprintf('CPD : %.2f, %.2f \n',mean(rmse_test(1,:)), std(rmse_test(1,:)));
