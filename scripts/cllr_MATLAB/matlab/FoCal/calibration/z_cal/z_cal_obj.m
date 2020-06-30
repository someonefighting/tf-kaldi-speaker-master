function obj = z_cal_obj(params,tar_scores,nontar_scores,lambda);
% this function is used as the optimization objective by TRAIN_CAL.M


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

% lambda*dy is a regularization penalty. If lambda is zero, the minimization may be 
% ill-conditioned.
obj1 = cllr(tar_llr,nontar_llr);
obj = obj1 + lambda*dy; 
fprintf('obj = %f = ',obj);fprintf('%f + ',obj1);fprintf('%f * ',lambda);fprintf('%f\n',dy);