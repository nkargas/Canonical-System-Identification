function [cvals, h_prod] = X_at(X, ind)

s = size(ind,1);
csubs = ind;

[~, F] = size(X{1});

N = length(X);
cvals = zeros(s, F);
for f = 1:F
    tvals = ones(s,1);
    for n = 1:N
        v = X{n}(:, f);
        tvals = tvals.*v(csubs(:, n));
    end
    cvals(:, f) = tvals;
end
h_prod = cvals;
cvals = sum(cvals, 2);
end
