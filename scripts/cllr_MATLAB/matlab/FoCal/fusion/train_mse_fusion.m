function w = train_mse_fusion(targets,non_targets,prior,w);
%  Train linear fusion with prior-weighted mean-squared-error objective.
%
%  Usage:
%    w = TRAIN_MSE_FUSION(targets,non_targets,prior,weights)
%    w = TRAIN_MSE_FUSION(targets,non_targets,prior)
%    w = TRAIN_MSE_FUSION(targets,non_targets)
%
%  Tip: For faster convergence, it is a good idea to first find a starting point via 
%       logistic regression:
%    w0 = TRAIN_LLR_FUSION(target,non_targets);
%    w = TRAIN_MSE_FUSION(targets,non_targets,prior,w0);
%   
%  Input parameters:
%    targets:     a [d,nt] matrix of nt target scores for each of d systems to be fused. 
%    non_targets: a [d,nn] matrix of nn non-target scores for each of the d systems.
%    prior:       (optional, default = 0.5), a scalar parameter between 0 and 1. 
%                 This weights the objective function, by replacing the effect 
%                 that the proportion nt/(nn+nt) has on the objective.
%                 For general use, omit this parameter (i.e. prior = 0.5).
%                 For NIST SRE, use: prior = effective_prior(0.01,10,1);
%    w:           (optional, default = zeros(d+1,1)), the starting point for minimization,
%                 w must be a column vector of size d+1.
%
%  Output parameters:
%    w: a vector of d+1 fusion coefficients. 
%         The first d coefficients are the weights for the d input systems.
%         The last coefficient is an offset (see below).
%
%  Fusion of new scores:
%    see: LIN_FUSION.m
%                                 
%

%  This code is an adapted version of the m-file 'train_cg.m' as made available by Tom Minka
%  at http://www.stat.cmu.edu/~minka/papers/logreg/.
%  Changes include:
%  - Different interface to the function.
%  - Omission of regularization penalty.
%  - Normalization and prior-weighting.
%  - Replacement of the logistic regression objective by the MSE objective.

if (nargin<3)
   prior = 0.5;
end;   

nt = size(targets,2);
nn = size(non_targets,2);
prop = nt/(nn+nt);
weights = [(prior/prop)*ones(1,nt),((1-prior)/(1-prop))*ones(1,nn)];


x = [[targets;ones(1,nt)],-[non_targets;ones(1,nn)]];
offset = logit(prior)*[ones(1,nt),-ones(1,nn)];

if (nargin<4)
  w = zeros(size(x,1),1);
end;


[d,n] = size(x);
old_g = zeros(size(w));
for iter = 1:1000
  old_w = w;
   
  arg = w'*x+offset;  % argument of sigma
  s = sigmoid(arg);   % sigma
  s1 = sigmoid(-arg); % 1-sigma
  
  g = -2*x*(s1.*s1.*s.*weights)'; % modified for MSE
  if iter == 1
    u = g;
  else
    u = cg_dir(u, g, old_g);
  end
  
  % line search along u
  ug = u'*g;
  ux = u'*x;
  a = 2*weights.*s1.*s1.*s.*(s1-2*s);    % modified for MSE
  uhu = (ux.^2)*a';
  w = w + (ug/uhu)*u;
  old_g = g;
  if max(abs(w - old_w)) < 1e-5
     break
  end
end
if iter == 1000
  warning('not enough iters')
end
