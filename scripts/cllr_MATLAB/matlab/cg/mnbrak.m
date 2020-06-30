function [ax,bx,cx] = mnbrak(f,ax,bx);

GOLD = 1.618034;
GLIMIT = 100;
TINY = 1.0e-20;

fa = feval(f,ax);
fb = feval(f,bx);

if (fb > fa)    % swap
   t = fa; fa = fb; fb = t;   
   t = ax; ax = bx; bx = t;   
end;

cx = bx+GOLD*(bx-ax);
fc = feval(f,cx);
while fb > fc
   r = (bx-ax)*(fb-fc);
   q = (bx-cx)*(fb-fa);
   u = bx-((bx-cx)*q-(bx-ax)*r)/(2*sign(q-r)*max(abs(q-r),TINY));
   ulim = bx+GLIMIT *(cx-bx);
   if (bx-u)*(u-cx) > 0
      fu = feval(f,u);
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
      fu = feval(f,u);
   elseif (cx-u)*(u-ulim) > 0
      fu = feval(f,u);
      if fu<fc
         bx = cx; cx=u; u = cx+GOLD*(cx-bx);
         fb = fc; fc = fu; fu = feval(f,u);
      end;   
   elseif (u-ulim)*(ulim*cx) >= 0
      u = ulim;
      fu = feval(f,u);
   else
      u = cx+GOLD*(cx-bx);
      fu = feval(f,u);
   end;   
   ax = bx; bx = cx; cx = u;
   fa = fb; fb = fc; fc = fu;
end;   