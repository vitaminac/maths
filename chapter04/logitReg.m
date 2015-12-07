function [w, llh] = logitReg(X, t, lambda)
% Logistic regression for binary classification optimized by Newton-Raphson
% method.
%   X: dxn data matrix
%   t: dx1 label (0/1)
%   lambda: regularization parameter
% Written by Mo Chen (sth4nth@gmail.com).
if nargin < 3
    lambda = 0;
end
X = [X; ones(1,size(X,2))];
[d,n] = size(X);


tol = 1e-4;
maxiter = 100;
llh = -inf(1,maxiter);

idx = (1:d)';
dg = sub2ind([d,d],idx,idx);
h = ones(1,n);
h(t==0) = -1;
w = rand(d,1);
z = w'*X;
for iter = 2:maxiter
    y = sigmoid(z);
    r = y.*(1-y);                       % 4.89
    Xw = bsxfun(@times, X, sqrt(r));
    H = Xw*Xw';                         % 4.95
    H(dg) = H(dg)+lambda;
    U = chol(H);
    g = X*(y-t)'+lambda.*w;
    p = -U\(U'\g);
    wo = w;
    while true      % line search
        w = wo+p;
        z = w'*X;   
        llh(iter) = -sum(log1pexp(-h.*z))-0.5*sum(lambda.*w.^2);
        incr = llh(iter)-llh(iter-1);
        if incr < 0
            p = p/2;
        else
           break;
        end
    end
    if incr < tol; break; end
end
llh = llh(2:iter);
% model.w = w(1:(end-1));
% model.w0 = w(end);