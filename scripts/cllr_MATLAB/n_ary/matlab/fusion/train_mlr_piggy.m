function W = train_mlr_piggy(features,llr,classification,lambda,W,prior);
%
%  Usage:
%    W = TRAIN_MLR_FUSION(features,classification,prior)
%    W = TRAIN_MLR_FUSION(features,classification)
%   
%  Input parameters:                                 
%    features: a [d,t] matrix of input features
%    classification: a [1,t] row of classifications for every input point. Values are in 1,2,...,n
%    prior: an n-ary prior to emphasize the objective 
%
%  Output:
%    W: a [d+1,n-1] matrix of fusion weights. The last row is an offset.
%    
%  Note: msigmoid(W'*[v;1]+mlogit(prior)) gives a linearly fused posterior for input vector v.

n = max(classification);

[d,t] = size(features);
features = [features;ones(1,t)];
d = d+1;


if nargin<5
   W = zeros(d,n-1);
end;   
w = W(:);

count = zeros(1,n);
s_it = zeros(n,t);
for i=1:n;
   ff = find(classification==i);
   count(i) = length(ff);
   s_it(i,ff)=ones(size(ff));
end;
prop = count/n;

if (nargin<4)
   lambda = zeros(1,d);
else 
   lambda = lambda*ones(1,d);
end;   
Lambda = diag(lambda);

if (nargin<6)
   prior = ones(n,1)/n;
end;   

prior = prior';

a_t = prior(classification)./prop(classification);
a_it = s_it.*(ones(n,1)*a_t);

af = features.*(ones(d,1)*a_t);

offset = mlogit(prior(:	))*ones(1,t);

old_g = zeros(size(w));
for iter = 1:(n-1)*1000
  old_w = w;
  % Sigma is the posterior
  Sigma = msigmoid(W'*features+offset+llr);
  G = af*Sigma'-features*a_it';
  G = G(:,1:n-1)+Lambda*W; 
  g = G(:);
  if iter == 1
    u = g;
  else
    u = cg_dir(u, g, old_g);
  end
  U = reshape(u,d,n-1);
  LU = Lambda*U;
  lu = LU(:);
  
  % line search along u
  ug = u'*g;
  uhu = lu'*u;
  for i=1:t;
     fU = features(:,i)'*U;
     uhu = uhu+( (fU*Sigma(1:n-1,i))^2 - fU*fU' ) * a_t(i);   
  end;   
  w = w + (ug/uhu)*u;
  W(:) = w;
  
  c = sum(sum(Lambda*(W.^2)))/2+sum(sum(-log(msigmoid(W'*features+offset+llr)).*a_it));
  fprintf('%i : %f\n',iter,c);
  
  old_g = g;
  if max(abs(w - old_w)) < 1e-7
     break
  end
end
if iter == 1000
  warning('not converged')
end

