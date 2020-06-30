function w = train_scal_cg(targets,non_targets,prior);

if (nargin<3)
   prior = 0.5;
end;   

nt = size(targets,2);
nn = size(non_targets,2);
prop = nt/(nn+nt);
weights = [(prior/prop)*ones(1,nt),((1-prior)/(1-prop))*ones(1,nn)];


x = [targets,-non_targets];
y = [ones(1,nt),-ones(1,nn)];
offset = logit(prior)*[ones(1,nt),-ones(1,nn)];

w = [0;0;5;-5];
n = nt+nn;
for iter = 1:1000
   
   
   a = exp(w(1));
   b = w(2);
   p = logit(w(3));
   q = logit(w(4));

   S = exp(a*x+b);
   pi = 1./(p*S+1-p);
   theta = 1./(q*S+1-q);
   L = log(theta)-log(pi);
   sigma = sigmoid(y.*L+offset);
   
   obj = log(sigma)*weights';
   fprintf('%d: ',iter); fprintf('obj = %f\n',obj);
   
   
   
   sig1 = 1-sigma;
   g0 = -sig1.*weights;
   g2 = g0.*S.*(p*pi-q*theta);
   g1 = g2.*x.*a;
   g3 = g0.*p*(1-p).*(S-1).*pi;
   g4 = -g0.*q*(1-q).*(S-1).*theta;
   G = [g1;g2;g3;g4];
   g = sum(G')';
   
   if iter == 1
     u = g;
   else
     u = cg_dir(u, g, old_g);
   end
   
   %d^2/d^2L 
   h0 = -g0.*sigma;
   H1 = repmat(h0,4,1).*G*G';
  
   ppiS = p*pi.*S;
   qthetaS= q*theta.*S;
   h22 = ppiS.*(1-ppiS)-qthetaS.*(1-qthetaS);
   h11 = h22.*(x.^2)*a^2;
   h12 = h22.*(x)*a;
   h23 = S.*(pi-p*(S-1).*(pi.^2))*p*(1-p);
   h24 = -S.*(theta-q*(S-1).*(theta.^2))*q*(1-q);
   h13 = h23.*x*a;  
   h14 = h24.*x*a;
   h33 = (S-1).*((1-2*p)*pi-p*(1-p)*(S-1).*(pi.^2))*p*(1-p);
   h44 = -(S-1).*((1-2*q)*theta-q*(1-q)*(S-1).*(theta.^2))*q*(1-q);
  
   h11 = g0*h11';
   h12 = g0*h12';
   h13 = g0*h13';
   h14 = g0*h14';
  
   h22 = g0*h22';
   h23 = g0*h23';
   h24 = g0*h24';
  
   h33 = g0*h33';
   h34 = 0;
  
   h44 = g0*h44';
  
   H2 = [h11 h12 h13 h14; h12 h22 h23 h24; h13 h23 h33 h34; h14 h24 h34 h44];
  
   H = H1+H2; 
  
   %fprintf('  g = %e: \n',g); fprintf('  H = %e\n',H);
   g,H,
  
  
   uHu = u'*H*u; 
   
   old_w = w;
   old_g = g;
   % line search along u
   ug = u'*g;
   w = w + (ug/uHu)*u;
   
   if max(abs(w - old_w)) < 1e-5
     break
   end
end

if iter == 1000
  warning('not enough iters')
end
