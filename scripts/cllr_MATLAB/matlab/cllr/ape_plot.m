function ape_plot(varargin);
%APE_PLOT: Applied-Probability-of-Error plot. 
%          Displays the goodness of log-likelihood-ratio detector outputs, 
%          for one or more detector systems.  
%
% Usage: APE_PLOT({'sys1',{tar_llrs1,nontar_llrs1}}, {'sys2',{tar_llrs2,nontar_llrs2}} ,...)   
%    
% Input parameters: One or more entries, one entry per system under evaluation.
%      'sys1': A string label for system1
%      tar_llrs1: An array of the target LLR outpouts of system1
%      nontar_llrs1: An array of the non-target LLR outpouts of system1.
%     
% Note: 'log' in log-likelihood-ratio, denotes the natural logarithm (base e).
%
% This routine generates two plots per system, APE-plot on top, and Cllr-plot below: 
% 
%   The top plot is the APE-plot, having three APE-curves. Each gives (y-axis) the  
%   error-rate of using the following three inputs to make Bayes decisions:
%   (a) Costs: Cmiss = Cfa = 1
%   (b) The target prior, given as 'logit prior' on the x-axis.
%   (c) log-likelihood-ratios, which are different for each plot:
%
%   	   RED: The 'actual' log-likelihood-ratios as supplied by the system under evaluation.
%        GREEN: Optimized log-likelihood-ratios as optimized by the evaluator (this code), 
%               on these given (evaluation) data. The optimization is subject only to a 
%               monotonicity constraint on the mapping between the actual and optimized 
%               llr values. 
%        BLACK(dashed): llrs = zeros. This forms the reference system which makes Bayes 
%                        decisions based on (a) and (b) only.
%        
%   Note 1: The APE-plot further shows a magenta dashed vertical line to indicate where the 
%           traditional NIST operating point is (at -2.29). The red and green error-rates 
%           at -2.29 are scaled versions of the traditional CDET and 'min CDET' values.
%   Note 2: The max of the green curve is also the EER.
%
%
% The bottom (Cllr) plot is a bar-graph giving (scaled) integrals under the green and red curves. 
% The (scaled) area under the reference (dashed black) curve is one. The integrals are performed 
% analytically over the whole x-axis (logit prior) from -inf to inf. The bar graph gives:
%   RED+GREEN: (Actual) total error-rate over the whole range of applications. 
%              This is CLLR(tar_llrs,nontar_llrs). 
%              Note: 0 <= Cllr <= inf
%   GREEN: (Minimum) total error-rate over the whole range of applications. This is the 
%          performance that the system under evaluation could have obtained with a 
%          perfect (for this data) score-to-llr calibration.
%          This is MIN_CLLR(tar_llrs,nontar_llrs). 
%          Note: 0 <= MIN_CLLR <= 1
%   RED:   This is the area between the red and the green APE-curves and is the measure of how
%          well the score to log-likelihood-ratio mapping is 'calibrated'. 
%          This is CLLR_CAL(tar_llrs,nontar_llrs).
%          Note: 0 <= CLLR_CAL <= inf


%
% Author: Niko Brummer, Spescom Datavoice.
% Disclaimer: This code is freely available for any non-commercial purpose, but the author and 
% his employer do not accept any responsibility for any consequences resulting from the use thereof.
% (E.g. getting an EER=50% at the NIST SRE.) 
%
% But if this code does prove useful, we would appreciate citation of the following article:
%   Niko Brummer and Johan du Preez, "Application-Independent Evaluation of Speaker Detection"
%   Computer Speech and Language, to be published, 2005. 



% number of systems
n=length(varargin); 

% plo = prior_log_odds
% This range was chosen to show more or less the range of interest for current levels of
% speaker detection performance. If systems get better and evaluation data more, this range 
% can be enlarged.
% The theoretcal range is the whole real line.
plo=-7:0.1:7;


clog=zeros(1,n);
minclog=zeros(1,n);
eer=0;
for i=1:n;
  data = varargin{i}; 
  name = data{1};
  llrs = data{2};
  scores = llrs;
  
  [tar_opt_llrs,nontar_opt_llrs] = opt_loglr(scores{1},scores{2},'raw');
  clog(i) = cllr(llrs{1},llrs{2});
  minclog(i) = cllr(tar_opt_llrs,nontar_opt_llrs);
  
  Pe = bayes_error_rate(llrs{1},llrs{2},plo);
  minPe = bayes_error_rate(tar_opt_llrs,nontar_opt_llrs,plo);
  refPe = bayes_error_rate(0,0,plo);
  subplot(2,n,i);plot(plo,minPe,'g',plo,Pe,'r',plo,refPe,'k--',[-2.29,-2.29],[0,1],'m--');grid;
  title(name);
  eer = max(eer,max(Pe));
end;

subplot(2,1,2);
colormap([0,0.7,0;1,0,0]);
if (n>1)
   bar([minclog;clog-minclog]','stacked');
else
   bar([[0,minclog,0];[0,clog-minclog,0]]','stacked');
end;
ylabel('C_{llr} [bits]');
set(gca,'xtick',[]);
set(gca,'xticklabel',[]);
legend('discrimination loss','calibration loss');
grid;


subplot(2,n,1);
ylabel('P(error)');
for i=1:n;
   subplot(2,n,i);
   xlabel('logit prior');
   axis([min(plo),max(plo),0,eer*1.1]);
end;   

