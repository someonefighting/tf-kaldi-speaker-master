function c = cllr(tar_llrs, nontar_llrs, prior);


lp = logit(prior);

% target trials 
c1 = mean(neglogsigmoid(tar_llrs+lp))/log(2);
  
% non_target trials 
c2 = mean(neglogsigmoid(-nontar_llrs-lp))/log(2);

% Cllr
c = prior*c1+(1-prior)*c2;
