function [x,val] = linmin(f,x0,x1,tol,maxiter,f0,f1);

if nargin<7
   [ax,bx,cx,fa,fb,fc] = mmnbrak(f,x0,x1);
else
   [ax,bx,cx,fa,fb,fc] = mmnbrak(f,x0,x1,f0,f1);
end;

dir = x1-x0;
[x,val] = mbrent(f,x0,dir,ax,bx,cx,tol,maxiter,fb);
x = x0 +x*dir;
