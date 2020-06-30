function p = msigmoid(y);
% p = MSIGMOID(y)
% Multi-sigmoid function. This is the inverse to MLOGIT.
% 
% Input: y is a real vector with n-1 components.
%
% Output: p is an n-ary discrete probability distribution of which 
%           the n (positive) components sum to one.


%pp = exp(y)/(1+sum(exp(y(:))));
%p=[pp(:);1-sum(pp(:))];
e = [exp(y);ones(1,size(y,2))];
p = e./(ones(size(y,1)+1,1)*sum(e));