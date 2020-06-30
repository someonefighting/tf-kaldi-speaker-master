function lp = logit(p)
% LOGIT: logit function.
%        This is a one-to-one mapping from probability to log-odds.
%        I.E. it maps the interval (0,1) to the real line.
%        The inverse function is given by SIGMOID.
%
%   log_odds = logit(p) = log(p/(1-p))

lp = log(p./(1-p));