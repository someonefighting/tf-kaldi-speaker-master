function J = blogit_jacobian(p);
n = length(p);
J = centering(n)*diag(p.^(-1));