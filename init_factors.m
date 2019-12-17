function [X] = init_factors(F, I, varargin)
params = inputParser;
params.addParameter('init', 'gaussian');

params.parse(varargin{:});
init = params.Results.init;

N = length(I);
A = cell(N, 1);
if strcmp(init,'gaussian')
    for n=1:N
        A{n} = randn(I(n), F);
    end
elseif strcmp(init,'uniform')
    for n=1:N
        A{n} = 0.5 + rand(I(n),F);
    end
end
X = ktensor(A);
