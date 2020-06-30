function scores = detect_all(llr,class);
[n,t]=size(llr);
n = n+1;
for i=1:n+1;
  scores{i} = [];   
end;   
for i=1:n;
   prior = 0.5*ones(n,1)/(n-1);
   prior(i) = 0.5;
   mlp = log(prior); mlp = mlp(1:end-1)-mlp(end);
   plo = llr+mlp*ones(1,t);
   %plo = llr;
   ee = exp([plo;zeros(1,t)]); post = ee(i,:) ./ sum(ee);
   f=find(class==i);
   scores{1} = [scores{1},post(f)];
   
   for j=1:i-1:i+1:n;
     f=find(class==j);
     scores{1+j} = [scores{1+j},post(f)];
   end;   
   
end;