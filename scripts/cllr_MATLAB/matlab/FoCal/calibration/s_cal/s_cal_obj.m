function obj = s_cal_obj(params,tar_scores,nontar_scores,lambda);
% this function is used as the optimization objective by TRAIN_S_CAL.M


a=exp(params(1));
b=params(2);
alpha = params(3);
beta = params(4);
dy = abs(alpha-beta);


tar_llr = s_cal(a,b,alpha,beta,tar_scores);
nontar_llr = s_cal(a,b,alpha,beta,nontar_scores);

% lambda*dy is a regularization penalty. If lambda is zero, the minimization may be 
% ill-conditioned.
obj1 = cllr(tar_llr,nontar_llr);
obj = obj1 + lambda*dy; 
fprintf('obj = %f = ',obj);fprintf('%f + ',obj1);fprintf('%f * ',lambda);fprintf('%f\n',dy);
%fprintf('alpha = %f;',alpha);fprintf(' beta = %f\n',beta);

