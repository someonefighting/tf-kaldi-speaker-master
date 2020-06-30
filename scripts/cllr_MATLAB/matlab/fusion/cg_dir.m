function dir = cg_dir(old_dir, grad, old_grad)
% Compute the new conjugate direction.

g = grad;
grad = grad(:);
old_grad = old_grad(:);

delta = grad - old_grad;
den = old_dir'*delta;
if (den==0)
   dir = g*0;
else   
   
  % Hestenes-Stiefel
  beta = (grad'*delta) / den;

  % Polak-Ribiere
  %beta = -grad'*(grad - old_grad) / (old_grad'*old_grad);
  
  % Fletcher-Reeves
  %beta = -(grad'*grad) / (old_grad'*old_grad);
  
  dir = g - beta*old_dir;
  
end;
