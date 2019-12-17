function [y,U] = get_y_U(Y, X, F, n, i, b)

N = ndims(X);
[r, ~] = find(Y.subs(:, n)==i);
y = (Y.vals(r) - b);
U = ones(length(r), F);

for nn = [1:n-1 n+1:N]
    U = U.*X.U{nn}(Y.subs(r, nn), :);
end
end