function [xa,xb,xc,fa,fb,fc,sa,sb,sc] = mnbrak(f,xa,xb,fa,fb,sa,sb);

GOLD = 1.618034;
GLIMIT = 100;
TINY = 1.0e-20;

%[fa,sa] = feval(f,xa,state);
%[fb,sb] = feval(f,xb,state);

if (fb > fa)    % swap
   t = fa; fa = fb; fb = t;   
   t = xa; xa = xb; xb = t;   
   t = sa; sa = sb; sb = t;   
end;

xc = xb+GOLD*(xb-xa);
[fc,sc] = feval(f,xc,state);
while fb > fc
   r = (xb-xa)*(fb-fc);
   q = (xb-xc)*(fb-fa);
   xu = xb-((xb-xc)*q-(xb-xa)*r)/(2*sign(q-r)*max(abs(q-r),TINY));
   ulim = xb+GLIMIT *(xc-xb);
   if (xb-xu)*(xu-xc) > 0
      [fu,su] = feval(f,xu,state);
      if fu<fc
         xa = xb; xb = xu;
         fa = fb; fb = fu;
         sa = sb; sb = su;
         return;
      elseif fu > fb
         xc = xu;
         fc = fu;
         sc = su;
         return;
      end;
      xu = xc+GOLD*(xc-xb);
      [fu,su] = feval(f,xu,state);
   elseif (xc-xu)*(xu-ulim) > 0
      [fu,su] = feval(f,xu,state);
      if fu<fc
         xb = xc; xc=xu; xu = xc+GOLD*(xc-xb);
         sb = sc; sc = su;
         fb = fc; fc = fu; 
         [fu,su] = feval(f,xu,state);
      end;   
   elseif (xu-ulim)*(ulim*xc) >= 0
      xu = ulim;
      [fu,su] = feval(f,xu,state);
   else
      xu = xc+GOLD*(xc-xb);
      [fu,su] = feval(f,xu,state);
   end;   
   xa = xb; xb = xc; xc = xu;
   fa = fb; fb = fc; fc = fu;
   sa = sb; sb = sc; sc = su;
   
end;   