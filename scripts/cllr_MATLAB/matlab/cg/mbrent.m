function [x,fx] = mbrent(f,x0,dir,ax,bx,cx,tol,maxiter,fb);

CGOLD = 0.3819660;
ZEPS = 1.0E-20;

e = 0;

if cx > ax
   a = ax;
   b = cx;
else
   a = cx;
   b = ax;
end;

x = bx; v = x; w = x;
fx = fb; fv = fx; fw = fx;

for iter=1:maxiter;
   xm = (a+b)/2;
   tol1 = tol*abs(x)+ZEPS;
   tol2 = 2*tol1;
   if abs(x-xm) <= tol2-(b-a)/2
      return;
   end;
   if abs(e) > tol1
      r = (x-w)*(fx-fv);
      q = (x-v)*(fx-fw);
      p = (x-v)*w-(x-w)*r;
      q = 2*(q-r);
      if q>0, p = -p; end;
      q = abs(q);
      etemp = e;
      e = d;
      if ( abs(p) > abs(0.5*q*etemp) ) |  (p<= q*(a-x)) | (p >= q*(b-x))
         if x>= xm; e = a-x; else e = b-x; end;
         d = CGOLD*e;
      else
         d = p/q;
         u = x+d;
         if (u-a<tol2) | (b-u < tol2); d = abs(tol1)*sign(xm-x); end;
      end;   
   else
      if x>=xm; e = a-x; else e = b-x; end;
      d = CGOLD*e;   
   end;
   if abs(d) >=tol1;
      u = x+d;
   else
      u = x+abs(tol1)*sign(d);
   end; 
   fu = feval(f,x0+dir*u);
   if fu <= fx;
      if (u>=x); a = x; else b = x; end;
      v = w; w = x; x = u;
      fv = fw; fw = fx; fx = fu;
   else
      if (u<x); a = u; else b=u; end;
      if (fu<=fw) | (w==x)
         v = w; w = u; 
         fv = fw; fw = fu;
      elseif (fu <= fv) | (v==x) | (v == w)
         v = u;
         fv = fu;
      end;
      
   end;
   
end;

% too many iterations
