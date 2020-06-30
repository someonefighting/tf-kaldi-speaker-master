function x = nlcg(f,x,step1,imax,jmax,eps1,eps2);

i = 0;
k = 0;

fgx = feval(f,x,1);
fx = fgx(1);
gx = fgx(2:end);

xold = x+step1*gx;

r = -gx;
d = r;
delta_new = r'*r;
delta0 = delta_new;
while i<imax and delta_new > eps1^2*delta0
   
   dist = sqrt(sum((x_old-x).^2));
   xtest = x+dist*r;
   [x,val] = linmin(f,x,xtest,tol,maxiter,f0,f1)
   
   delta_old = delta_new;
   delta_new = r'*r;
   beta = delta_new/delta_old;
   d = r + beta*d;
   k = k + 1;
   if (k==n) | (r'd<=0)
      d = r;
      k = 0;
   end;
   i = i+1;
end;   
      