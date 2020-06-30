function W = train_mlr_kernel(Gram,X,llr,classification,lambda,maxiter,W);


n = max(classification);

features = [Gram;X];
[d,t] = size(features);

if nargin<7
   W = zeros(d,n-1);
end;   
w = W(:);

if nargin<6
   maxiter = 100;
end;   

alpha = 1:t;
beta = t+1:d;

count = zeros(1,n);
s_it = zeros(n,t);
for i=1:n;
   ff = find(classification==i);
   count(i) = length(ff);
   s_it(i,ff)=ones(size(ff));
end;
prop = count/n;

prior = ones(n,1)/n;
prior = prior';

a_t = prior(classification)./prop(classification);
a_it = s_it.*(ones(n,1)*a_t);

af = features.*(ones(d,1)*a_t);

offset = mlogit(prior(:	))*ones(1,t);

old_g = zeros(size(w));
for iter = 1:maxiter
  old_w = w;
  
  alphaK = W(alpha,:)'*Gram;
  betaX = W(beta,:)'*X;
  
  % Sigma is the posterior
  Sigma = msigmoid(llr+alphaK+betaX+offset);
  G = af*Sigma'-features*a_it';
  G = G(:,1:n-1);
  G(alpha,:) = G(alpha,:) + lambda*alphaK';
  g = G(:);
  if iter == 1
    u = g;
  else
    u = cg_dir(u, g, old_g);
  end
  U = reshape(u,d,n-1);
  
  % line search along u
  ug = u'*g;
  uhu = 0;
  for i=1:n-1;
    uhu = uhu + lambda*U(alpha,i)'*Gram*U(alpha,i);   
  end;   
  for i=1:t;
     fU = features(:,i)'*U;
     uhu = uhu+( (fU*Sigma(1:n-1,i))^2 - fU*fU' ) * a_t(i);   
  end;   
  w = w + (ug/uhu)*u;
  W(:) = w;
  
  pen = 0;
  for i=1:n-1;
    pen = pen + alphaK(i,:)*W(alpha,i);
  end;
  c = sum(sum(-log(msigmoid(llr+W'*features+offset)).*a_it));
  fprintf('%i : %f = %f + %f\n',iter,c+lambda*pen/2,c,+lambda*pen/2);
  
  old_g = g;
  if max(abs(w - old_w)) < 1e-12
     break
  end
end
