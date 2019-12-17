function [X, b, Out] = csid(inputs, targets, F, I, f_type, opts, varargin)

params = inputParser;
params.addParameter('reg_fro', 0, @(x) x >= 0);
params.addParameter('reg_smooth', 0, @(x) x >= 0);
params.addParameter('max_itr', 500, @(x) isscalar(x) & x > 0);
params.addParameter('tol', 1e-4, @(x) x > 0);
params.addParameter('printitn', 10);
params.addParameter('step_size', 1e-4);
params.addParameter('bias', true);
params.parse(varargin{:});

mu = params.Results.reg_fro;
mu_sm = params.Results.reg_smooth;
max_itr = params.Results.max_itr;
tol = params.Results.tol;
printitn = params.Results.printitn;
bias = params.Results.bias;
[~, N] = size(inputs);

n_outputs = size(targets, 2);
if n_outputs > 1
    I = [I n_outputs];
    N = N + 1;
end

if bias == true
    b = mean(targets(opts.tr_ind, :), 1);
    b = mean(b);
else
    b = 0;
end

% Initialization
X = init_factors(F, I,'init', 'uniform');

% Regularization matrices
T = cell(N, 1);
T_T = cell(N, 1);
for n = 1:N
    if f_type(n) == 0
        T{n} = toeplitz([1; zeros(I(n)-2, 1)] , [1 -1 zeros(1, I(n)-2)]); 
        %T{n} = toeplitz([1; zeros(I(n)-3, 1)] , [1 -2 1 zeros(1, I(n)-3)]);
    else
        T{n} = zeros(1, I(n));
    end
    T_T{n} = T{n}'*T{n};
end

Out.cost = zeros(max_itr, 1);
inputs_tr = inputs(opts.tr_ind, :); s_tr = length(opts.tr_ind); targets_tr = targets(opts.tr_ind, :);
inputs_vl = inputs(opts.vl_ind, :); s_vl = length(opts.vl_ind); targets_vl = targets(opts.vl_ind, :);
inputs_te = opts.inputs_te; s_te = size(opts.inputs_te, 1); targets_te = opts.targets_te;

Out.rmse_tr = zeros(max_itr,1);
Out.rmse_vl = zeros(max_itr,1);
Out.rmse_te = zeros(max_itr,1);

Y.subs = inputs_tr;
Y.vals = targets_tr;

iter = 1;
best_X = X;
best_b = b;
best_rmse = inf;
cnt_rmse = 0;
thresh = 5;

while(1)
    %Update parameters
    for n=1:N
        for i = 1:I(n)
            [y, U] = get_y_U(Y, X, F, n, i, b);
            X.U{n}(i, :) = (1/s_tr * (U'*U) + mu*eye(F) + mu_sm*T_T{n}(i,i)*eye(F) )\(1/s_tr*U'*y - mu_sm*X.U{n}([1:i-1 i+1:end], :)'*T_T{n}(i, [1:i-1 i+1:end])');
        end
    end
    if bias == true
        b = mean2(targets_tr - X_at(X, inputs_tr));
    end
    
    % predictions
    [pred_tr] = X_at(X, inputs_tr) + b;
    [pred_vl] = X_at(X, inputs_vl) + b;
    [pred_te] = X_at(X, inputs_te) + b;
    
    Out.rmse_tr(iter) = sqrt((1/(s_tr*n_outputs) * norm(pred_tr - targets_tr, 'fro')^2));
    Out.rmse_vl(iter) = sqrt((1/(s_vl*n_outputs) * norm(pred_vl - targets_vl, 'fro')^2));
    Out.rmse_te(iter) = sqrt((1/(s_te*n_outputs) * norm(pred_te - targets_te, 'fro')^2));
    Out.cost(iter) = compute_cost(Out.rmse_tr(iter)^2, N, X, T, mu, mu_sm);
    
    if iter > 1
        rel = abs(Out.cost(iter) - Out.cost(iter-1))/abs(Out.cost(iter-1));
        
        if mod(iter, printitn) == 0
            fprintf('Relative change : %d \n', rel);
            fprintf('RMSE train : %f \n', Out.rmse_tr(iter));
            fprintf('RMSE valid : %f \n', Out.rmse_vl(iter));
            fprintf('RMSE test : %f \n', Out.rmse_te(iter));
            fprintf('Bias : %f \n', b);
        end
        
        if Out.rmse_vl(iter) < best_rmse
            best_rmse = Out.rmse_vl(iter);
            best_rmse_test = Out.rmse_te(iter);
            Out.pred_te = pred_te;
            
            best_X = X;
            best_b = b;
        end
        
        if Out.rmse_vl(iter)>Out.rmse_vl(iter-1)
            cnt_rmse = cnt_rmse + 1;
        else
            cnt_rmse = 0;
        end
        
        if iter == max_itr || rel < tol || cnt_rmse >= thresh
            fprintf('RMSE test : %f, RMSE valid : %f, Rank : %d mu : %f mu_sm : %f \n', best_rmse_test, best_rmse, F, mu, mu_sm);
            Out.cost(iter+1:end) = [];
            Out.rmse_tr(iter+1:end) = [];
            Out.rmse_vl(iter+1:end) = [];
            Out.best_rmse = best_rmse;
            X = best_X;
            b = best_b;
            Out.test_rmse = best_rmse_test;
            break;
        end
    end
    iter = iter + 1;
end

end

function er = compute_cost(mse, N, X, T, mu, mu_sm)
er = 0;
for n = 1:N
    er = er + mu*norm(X.U{n}(:))^2;
    er = er + mu_sm*norm(T{n}*X.U{n},'fro')^2;
end
er = er +  mse;
end
