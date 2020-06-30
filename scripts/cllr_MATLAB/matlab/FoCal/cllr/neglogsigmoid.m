function neg_log_p = neglogsigmoid(log_odds);

% neg_log_p = NEGLOGSIGMOID(log_odds)
%   This is mathematically equivalent to -log(sigmoid(log_odds)), but possibly numerically better.          

neg_log_p = -log_odds;
e = exp(-log_odds);
f=find(e<e+1);
neg_log_p(f) = log(1+e(f));
