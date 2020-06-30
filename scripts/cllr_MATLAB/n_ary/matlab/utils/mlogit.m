function y = mlogit(p);
% y = MLOGIT(p)
% Multi-logit function.
% 
% Input: p is a vector, representing a multinomial probability distribution. 
%        Conditions: length(p)>1, sum(p(:)) = 1, min(p) > 0 and max(p) < 1.

% Output: y is a real vector, where length(y) = (length(p)-1)


last = size(p,1);
y = zeros(last-1,size(p,2));
for i=1:last-1;
   y(i,:) = log(p(i,:)./p(last,:));
end;   