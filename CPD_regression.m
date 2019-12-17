function md = CPD_regression(opts)

p_val = 0.2; % Early stopping

I = opts.I;
f_type = opts.f_type;
[F_, mu_, mu_smooth_] = ndgrid(opts.F, opts.mu, opts.mu_smooth);

comb_param = [F_(:), mu_(:), mu_smooth_(:)];
[n_samples, N] = size(opts.inputs_tr);

% Discretize everything
[inputs, I, ~] = N_discretize([opts.inputs_tr; opts.inputs_te], I, N, f_type, 'kmeans');

inputs_tr = inputs(1:n_samples, :);
targets_tr = opts.targets_tr;
inputs_te = inputs(n_samples + 1:end, :);

X = cell(size(comb_param, 1), 1);
b_out = cell(size(comb_param, 1), 1);
valid_rmse = zeros(size(comb_param, 1), 1);
test_rmse = zeros(size(comb_param, 1), 1);

% Keep p_val as a validation set
tol = opts.tol;
max_itr = opts.max_itr;
b = opts.b;
opts_inner.inputs_te = inputs_te;
opts_inner.targets_te = opts.targets_te;
indices = randperm(n_samples);
opts_inner.tr_ind = indices(1:floor((1-p_val)*n_samples));
opts_inner.vl_ind = indices(floor((1-p_val)*n_samples)+1:end);

pred_te = cell(size(comb_param, 1), 1);
parfor i=1:size(comb_param, 1)
    [X{i}, b_out{i}, Out] = csid(inputs_tr, targets_tr, comb_param(i, 1), I, ...
      f_type, opts_inner, 'reg_fro', comb_param(i, 2), 'reg_smooth', comb_param(i, 3), ...
      'max_itr', max_itr, 'tol', tol, 'bias', b, 'printitn', 200);
    valid_rmse(i) = Out.best_rmse;
    test_rmse(i) = Out.test_rmse;
    pred_te{i} = Out.pred_te;
end

md.X = X;
md.b_out = b_out;
md.valid_rmse = valid_rmse;
[~, ind] = min(md.valid_rmse);
md.test_rmse = test_rmse(ind);
md.pred_te = pred_te{ind};
end

function [inputs, I, partition] = N_discretize(inputs, d_int, N, f_type, d_type)
fprintf('K-means clustering \n')
I = zeros(1, N);
partition = cell(N, 1);
switch d_type
    case 'kmeans'
        for n=1:N
            ind_non_nan = ~isnan(inputs(:, n));
            if ~f_type(n)
                n_uniq = length(unique(inputs(ind_non_nan, n)));
                if n_uniq > d_int
                    [~, C] = kmeans(inputs(ind_non_nan, n), d_int, 'Replicates', 10, 'MaxIter', 1000);
                    [partition{n}, ~, ~] = lloyds(inputs(ind_non_nan, n), sort(C, 'ascend'));
                    inputs(ind_non_nan, n) = quantiz(inputs(ind_non_nan, n), partition{n}) + 1;
                else
                    inputs(ind_non_nan ,n) = knnsearch(sort(unique(inputs(ind_non_nan, n)), 'ascend'), inputs(ind_non_nan, n));
                end
            end
            I(n) = length(unique(inputs(ind_non_nan, n)));
        end
end
fprintf('K-means clustering done \n')
end
