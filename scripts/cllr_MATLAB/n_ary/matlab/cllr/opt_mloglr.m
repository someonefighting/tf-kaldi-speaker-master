function mloglr = opt_mloglr(scores,class,option);

n = max(class);

mloglr = scores;

for i=1:n-1;
   tar_scores = scores(i,find(class==i));
   nontar_scores = scores(i,find(class~=i));
   
   [tar_llrs,nontar_llrs] = opt_loglr(tar_scores,nontar_scores,option);
   
   mloglr(i,find(class==i)) = tar_llrs ;
   mloglr(i,find(class~=i)) = nontar_llrs ;
    
end;

