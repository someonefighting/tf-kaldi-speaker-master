function [a,b,alpha,beta] = train_s_cal(tar_scores,nontar_scores,lambda);
% [a,b,alpha,beta] = TRAIN_S_CAL(tar_scores,nontar_scores,score_label)
% [a,b,alpha,beta] = TRAIN_S_CAL(tar_scores,nontar_scores,score_label,lambda)
%
%   -Optimizes the parameters of the score-to-llr mapping 's_cal.m'.
%   -Plots the APE-plots of the given scores and the optimized LLRs.
%
%   Input parameters:
%     tar_scores, nontar-scores: arrays of scores that act as training data to optimize this 
%                                calibration mapping
%     lambda: a scalar or an array (of small postive values), to be used as a regularization 
%             penalty.
%             If lambda is an array, the optimization is repeated for every penalty. 
%             Decreasing the regularization penalty in two or three steps may be a good idea.
%             Default: lambda = [ 0.1  0.01  0.001 ]
%
%   Output:
%     [a,b,alpha,beta]: Optimized parmaters that can be used as is in the mapping 
%     function scal.m (to map new scores to LLRs).


if (nargin<3)
   lambda = [0.01, 0.001];
end;   

params = [0,0,5,-5];
for i=1:length(lambda)
  params = fmins('s_cal_obj',[0,0,5,-5],[],[],tar_scores,nontar_scores,lambda(i));
end;

a=exp(params(1));
b=params(2);
alpha = params(3);
beta = params(4);

tar_llr = s_cal(a,b,alpha,beta,tar_scores);
nontar_llr = s_cal(a,b,alpha,beta,nontar_scores);

fprintf('calculating ape plots ...');

ape_plot({'raw',{tar_scores,nontar_scores}},{'scal',{tar_llr,nontar_llr}});

fprintf(' done\n');

