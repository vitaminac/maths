function [w, llh] = logitReg(X, t, lambda)
% Logistic regression for binary classification (Bernoulli likelihood)
% Written by Mo Chen (sth4nth@gmail.com).
if nargin < 3
    lambda = 1e-2;
end
X = [X; ones(1,size(X,2))];
[w, llh] = optNewton(X, t, lambda, zeros(size(X,1),1));