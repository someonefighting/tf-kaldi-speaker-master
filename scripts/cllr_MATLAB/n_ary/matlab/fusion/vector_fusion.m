function f = vector_fusion(weights,features);
%  Performs linear fusion of multiple rows of detection scores
%
%  Usage:
%    f = LIN_FUSION(weights,features)
%
%  Input parameters:
%    weights: a [d+1,n] matrix of fusion weights. The first d rows are multiplied 
%             with the input features, the final row is an offset. 
%    scores:  a [d,t] matrix of d features for each of t trials.
%
%  Output:
%    f: an [n,t] matrix .
%

f = weigts'*[features;ones(1,size(scores,2))];