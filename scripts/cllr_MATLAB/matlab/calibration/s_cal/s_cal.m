function llr = s_cal(a,b,alpha,beta,scores);
% llr = S_CAL(a,b,alpha,beta,scores)
%
%   Maps score to log-likelihood-ratio via linear mapping, followed by a sigmoid saturation:
%
%   - The linear part is defined by the slope, a and the offset, b:
%        y = a*scores + b
%
%   - The sigmoid part (see the code for the definition) is parameterized by real scalars 
%     alpha and beta:
%     In normal use, alpha > beta. In this case, the sigmoid mapping is 
%     monotonically rising. But if alpha = beta, the mapping is flat (constant), and if
%     alpha < beta, then the mapping is monotonically decreasing. (This calibration can 
%     therefore even compensate for a score of the wrong sign!)
%
%     If alpha >>0 and beta <<0, then the sigmoid saturates at a minimum of -alpha and 
%     a maximum of -beta.
%
%   Output: 
%     llr: is an array of the same size as scores
%

% linear mapping
y = a*scores+b;

% sigmoid saturation
p = sigmoid(alpha);
q = sigmoid(beta);
r = exp(y);
llr = log((p*(r-1)+1)./(q*(r-1)+1));
