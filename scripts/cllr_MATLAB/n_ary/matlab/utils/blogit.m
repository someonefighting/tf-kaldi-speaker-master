function y = blogit(p);

lp = log(p);
m = mean(lp);
y = zeros(size(p));
for i=1:size(p,1)
   y(i,:) = lp(i,:)-m;
end;   