function y = logMvGamma(x,d)
% Compute logarithm multivariate Gamma function.
% Gamma_p(x) = pi^(d(d-1)/4) \prod_(j=1)^d Gamma(x+(1-j)/2)
% log(Gamma_p(x)) = d(d-1)/4 log(pi) + \sum_(j=1)^d log(Gamma(x+(1-j)/2))
% Written by Michael Chen (sth4nth@gmail.com).
s = size(x);
x = reshape(x,1,prod(s));
x = bsxfun(@plus,repmat(x,d,1),(1-(1:d)')/2);
y = d*(d-1)/4*log(pi)+sum(gammaln(x),1);
y = reshape(y,s);