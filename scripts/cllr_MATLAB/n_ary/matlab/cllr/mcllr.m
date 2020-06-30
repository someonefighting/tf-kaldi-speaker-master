function c = mcllr(llr,class);

c = 0;
n = max(class);
post = msigmoid(llr);
for i = 1:n;
   c = c -mean(log(post(i,find(class==i))))/(n*log(n));   
   
end;