function llr = lda_llr2(T,scores);
llr = T*[scores;ones(1,size(scores,2))];