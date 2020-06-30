function w = train_llr_fusion(targets,non_targets,prior);
%  Train Linear fusion with prior-weighted Logistic Regression objective.
%  The fusion output is encouraged by this objective to be a well-calibrated log-likelihood-ratio.
%  I.E., this is simultaneous fusion and calibration.
%
%  Usage:
%    w = TRAIN_LLR_FUSION(targets,non_targets,prior)
%    w = TRAIN_LLR_FUSION(targets,non_targets)
%   
%  Input parameters:
%    targets:     a [d,nt] matrix of nt target scores for each of d systems to be fused. 
%    non_targets: a [d,nn] matrix of nn non-target scores for each of the d systems.
%    prior:       (optional, default = 0.5), a scalar parameter between 0 and 1. 
%                 This weights the objective function, by replacing the effect 
%                 that the proportion nt/(nn+nt) has on the objective.
%                 For general use, omit this parameter (i.e. prior = 0.5).
%                 For NIST SRE, use: prior = effective_prior(0.01,10,1);
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


if (nargin<3)
   prior = 0.5;
end;   

nt = size(targets,2);
nn = size(non_targets,2);
prop = nt/(nn+nt);
weights = [(prior/prop)*ones(1,nt),((1-prior)/(1-prop))*ones(1,nn)];


x = [[targets;ones(1,nt)],-[non_targets;ones(1,nn)]];
w = zeros(size(x,1),1);
offset = logit(prior)*[ones(1,nt),-ones(1,nn)];

[d,n] = size(x);
old_g = zeros(size(w));
for iter = 1:1000
  old_w = w;
  % s1 = 1-sigma
  s1 = 1./(1+exp(w'*x+offset));
  g = x*(s1.*weights)';
  if iter == 1
    u = g;
  else
    u = cg_dir(u, g, old_g);
  end
  
  % line search along u
  ug = u'*g;
  ux = u'*x;
  a = weights.*s1.*(1-s1);
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
