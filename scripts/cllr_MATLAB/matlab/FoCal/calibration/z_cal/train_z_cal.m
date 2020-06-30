function [xmin,ymin,xmax,ymax] = train_z_cal(tar_scores,nontar_scores,lambda);
% [xmin,ymin,xmax,ymax] = TRAIN_Z_CAL(tar_scores,nontar_scores,score_label)
% [xmin,ymin,xmax,ymax] = TRAIN_Z_CAL(tar_scores,nontar_scores,score_label,lambda)
%
%   -Optimizes the parameters of the score-to-llr mapping 'zcal.m'.
%   -Plots the APE-plots of the given scores and the optimized LLRs.
%
%   Input parameters:
%     tar_scores, nontar-scores: arrays of scores that act as training data to optimize this 
%                                calibration mapping
%     lambda: a scalar or an array (of small postive values), 
%             to be used as a regularization penalty.
%             If lambda is an array, the optimization is repeated once with every penalty. 
%             Decreasing the regularization penalty in two or three steps may be a good idea.
%             Default: lambda = [ 0.1  0.01  0.001 ]
%
%   Output:
%     [xmin,ymin,xmax,ymax]: Optimized parmaters that can be used as is in the mapping 
%     function zcal.m (to map new scores to LLRs).

if (nargin<3)
   lambda = [0.1, 0.01, 0.001];
end;   

%initial parameter setting
params = [0,0,0,0];

% minimize with decreasing regularization penalty
for i=1:length(lambda);
   params = fmins('z_cal_obj',params,[],[],tar_scores,nontar_scores,lambda(i));
end;   

x0=params(1);
y0=params(2);
dx = exp(params(3));
dy = exp(params(4));
xmin = x0-dx/2;
xmax = x0+dx/2;
ymin = y0-dy/2;
ymax = y0+dy/2;

tar_llr = z_cal(xmin,ymin,xmax,ymax,tar_scores);
nontar_llr = z_cal(xmin,ymin,xmax,ymax,nontar_scores);

fprintf('calculating ape plots ...');

ape_plot({'raw',{tar_scores,nontar_scores}},{'zcal',{tar_llr,nontar_llr}});

fprintf(' done\n');

