function p = sigmoid(log_odds);
% SIGMOID: Inverse of the logit function.
%          This is a one-to-one mapping from log odds to probability. 
%          I.E. it maps the real line to the interval (0,1).
%
%   p = sigmoid(log_odds)

p = 1./(1+exp(-log_odds));