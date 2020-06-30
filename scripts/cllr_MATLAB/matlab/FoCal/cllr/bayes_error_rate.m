function Pe = bayes_error_rate(tar_llrs, nontar_llrs, prior_log_odds);

% Pe = bayes_error_rate(tar_llrs, nontar_llrs, prior_log_odds);
%
%   Pe is the probability of error, when Bayes decisions are made with the following three inputs:
%   (a) Cmiss = Cfa = 1
%   (b) The given prior
%   (c) The log-likelihood-ratios, given separately here for target and non-target trials
%
%   Input parameters: 
%     tar_llrs: an array of log-likelihood-ratios for target trials
%     nontar_llrs: an array of log-likelihood-ratios for non-target trials
%     prior_log_odds: an array representing a range over the prior, specified as log-odds
%
%   Output: Pe is an array of the same size as prior_log_odds. I.E. each Pe value corresponds
%           to one value of the prior.

% Author: Niko Brummer, Spescom Datavoice.
% Disclaimer: This code is freely available for any non-commercial purpose, but the author and 
% his employer do not accept any responsibility for any consequences resulting from the use thereof.
% (E.g. getting an EER=50% at the NIST SRE.) 
%
% But if this code does prove useful, we would appreciate citation of the following article:
%   Niko Brummer and Johan du Preez, "Application-Independent Evaluation of Speaker Detection"
%   Computer Speech and Language, to be published, 2005. 



% These error-rates need no introduction. They vary as the decision threshold is adjusted. 
% In turn, the threshold is dependent on the prior.
pmiss = zeros(size(prior_log_odds));
pfa = zeros(size(prior_log_odds));

% loop over different priors
for i=1:length(prior_log_odds);
   
  % target trials 
  posterior = sigmoid(tar_llrs + prior_log_odds(i));
  pmiss(i) = mean((1-sign(posterior-0.5))/2); % error-rate of Bayes decisions
   
  % non_target trials 
  posterior = sigmoid(nontar_llrs + prior_log_odds(i));
  pfa(i) = mean((1-sign(0.5-posterior))/2); % error_rate of Bayes decisions


end;

% Combine with prior-weighting, and we're done.
Pe = pmiss.*sigmoid(prior_log_odds) +  pfa.*sigmoid(-prior_log_odds);
