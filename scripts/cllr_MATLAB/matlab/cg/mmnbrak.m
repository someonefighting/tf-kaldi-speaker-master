function [ax,bx,cx,fa,fb,fc] = mmnbrak(f,x0,x1,f0,f1);

GOLD = 1.618034;
GLIMIT = 100;
TINY = 1.0e-20;

if nargin<5
   f0 = feval(f,x0);
   f1 = feval(f,x1);
end;   

dir = x1-x0;
ax = 0;
bx = 1;


fa = f0;
fb = f1;

if (fb > fa)    % swap
   t = fa; fa = fb; fb = t;   
   t = ax; ax = bx; bx = t;   
end;

cx = bx+GOLD*(bx-ax);
fc = feval(f,x0+cx*dir);
while fb > fc
   r = (bx-ax)*(fb-fc);
   q = (bx-cx)*(fb-fa);
   u = bx-((bx-cx)*q-(bx-ax)*r)/(2*sign(q-r)*max(abs(q-r),TINY));
   ulim = bx+GLIMIT *(cx-bx);
   if (bx-u)*(u-cx) > 0
      fu = feval(f,x0+u*dir);
      if fu<fc
         ax = bx;
         bx = u;
         fa = fb;
         fb = fu;
         return;
      elseif fu > fb
         cx = u;
         fc = fu;
         return;
      end;
      u = cx+GOLD*(cx-bx);
      fu = feval(f,x0+u*dir);
   elseif (cx-u)*(u-ulim) > 0
      fu = feval(f,x0+u*dir);
      if fu<fc
         bx = cx; cx=u; u = cx+GOLD*(cx-bx);
         fb = fc; fc = fu; fu = feval(f,x0+u*dir);
      end;   
   elseif (u-ulim)*(ulim*cx) >= 0
      u = ulim;
      fu = feval(f,x0+u*dir);
   else
      u = cx+GOLD*(cx-bx);
      fu = feval(f,x0+u*dir);
   end;   
   ax = bx; bx = cx; cx = u;
   fa = fb; fb = fc; fc = fu;
end;   