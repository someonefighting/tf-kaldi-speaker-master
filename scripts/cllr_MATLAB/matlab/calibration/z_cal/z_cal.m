function llr = z_cal(xmin,ymin,xmax,ymax,scores);
% llr = Z_CAL(xmin,ymin,xmax,ymax,scores)
%   Maps score to log-likelihood-ratio via a clipped linear mapping.
%     The linear part is defined by the line-segment between the points 
%     (xmin,ymin) and (xmax,ymax) .
%     The mapping is clipped below at ymin and above at above at ymax.
%
%   The name "Z" is mnemonic for the shape of the mapping.
%
%   Input parameters:
%     xmin,xmax,ymin,ymax: are real numbers, such that xmax > xmin and ymax > ymin 
%     scores is an array of input scores
%
%   Output: 
%     llr: is an array of the same size as scores
%

m = (ymax-ymin)/(xmax-xmin);
c = ymin-m*xmin;
llr = m*scores+c;

llr(find(llr<ymin))=ymin;
llr(find(llr>ymax))=ymax;

