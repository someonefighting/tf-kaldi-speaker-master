function c = cllr_cal(tar_llrs, nontar_llrs);
% CLLR_CAL: Measure of log-likelihood-ratio calibration.
%
% 		c = CLLR_CAL(tar_llrs, nontar_llrs);
%
%     Input parameters:
%       tar_llrs: an array of log-likelihood-ratios for target trials
%       nontar_llrs: an array of log-likelihood-ratios for non-target trials
%
%     Note: 'log' in log-likelihood-ratio, denotes the natural logarithm (base e).
%
%     Range: 0 <= c <= inf. 
%     Sense: c << 1 is good, c close to 1 is bad, c >> 1 is very bad
%     Note: CLLR = MIN_CLLR + CLLR_CAL. 
%           Therefore, bad calibration can destroy all of the benefit of a good MIN_CLLR.

%
% Author: Niko Brummer, Spescom Datavoice.
% Disclaimer: This code is freely available for any non-commercial purpose, but the author and 
% his employer do not accept any responsibility for any consequences resulting from the use thereof.
% (E.g. getting an EER=50% at the NIST SRE.) 
%
% But if this code does prove useful, we would appreciate citation of the following article:
%   Niko Brummer and Johan du Preez, "Application-Independent Evaluation of Speaker Detection"
%   Computer Speech and Language, to be published, 2005. 

c = cllr(tar_llrs,nontar_llrs) - min_cllr(tar_llrs,nontar_llrs);
