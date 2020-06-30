function T = train_lda2(scores,class);
[m,t]=size(scores);
C = zeros(m,m);
n = max(class);
for i=1:n;
   C = C + cov(scores(:,find(class==i))',1)/n;   
end;

means = zeros(m,n);
for i=1:n;
   s=scores(:,find(class==i));
   means(:,i) = mean(s')';
end;


K = inv(C);
T = zeros(n-1,m+1);
lq = means(:,n)'*K*means(:,n);
for i=1:n-1;
   q = means(:,i)'*K*means(:,i);
   delta = means(:,i)-means(:,n);
   T(i,:)=[delta'*K,(lq-q)/2];
end;   