function p = bsigmoid(y);
% p = CSIGMOID(y)
% Multi-sigmoid function. This is the inverse to CLOGIT.
% 
% Input: y is a real vector with n components.
%
% Output: p is an n-ary discrete probability distribution of which 
%           the n (positive) components sum to one.


e = exp(y);
s = sum(e);
for i=1:size(y,1);
   p(i,:) = e(i,:)./s;
end;   