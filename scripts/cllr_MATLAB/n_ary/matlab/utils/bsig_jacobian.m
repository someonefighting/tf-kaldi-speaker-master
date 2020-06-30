function J = bsig_jacobian(p);
% J = BSIG_JACOBIAN(p)
% This is the Jacobian (matrix of first derivatives[ d p_i / d x_j ] ) of the 
% 'softmax' function:
%   pi = exp(xi)/ sum(exp(x_k))
% 
% J is symmetric and is positive semidefinite:
%   x'Jx = sum(p_i (x_i)^2 ) - (sum(p_i x_i))^2 >=0 by Jensen's inequality
%   but for x = alpha*ones(n,1): x'Jx = 0.
%
% J has rank n-1: one zero eigenvalue and the rest positive, as long as p_i > 0
%
% Note also J*centering(n) = J.

n = length(p);
J = diag(p)-p*p';