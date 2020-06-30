function f = lin_fusion(weights,scores);
%  Performs linear fusion of multiple rows of detection scores
%
%  Usage:
%    f = LIN_FUSION(weights,scores)
%
%  Input parameters:
%    weights: a d+1 column-vector of fusion weights. The first d weights are multiplied 
%             with the input scores, the final weight is an offset. 
%    scores:  a [d,n] matrix of n scores for each of d detection systems.
%
%  Output:
%    f: a row of n fused scores.
%

f = weights'*[scores;ones(1,size(scores,2))];