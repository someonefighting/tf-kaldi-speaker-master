function [R,means]=train_lda(scores,class);
[m,t]=size(scores);
C = zeros(m,m);
n = max(class);
for i=1:n;
   C = C + cov(scores(:,find(class==i))',1)/n;   
end;
R = inv(sqrtm(C));
scores = R*scores;

means = zeros(m,n);
for i=1:n;
   s=scores(:,find(class==i));
   means(:,i) = mean(s')';
end;
