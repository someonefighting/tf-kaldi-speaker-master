#!/bin/bash

data=$2
nnet_dir=$1

substract_mean() {
  data_name=$1
  nnet_dir=$2
  [ ! -f $nnet_dir/xvectors_$data_name/xvector.scp.ori ] && cp $nnet_dir/xvectors_$data_name/xvector.scp $nnet_dir/xvectors_$data_name/xvector.scp.ori
  cp $nnet_dir/xvectors_$data_name/xvector.scp $nnet_dir/xvectors_$data_name/xvector.scp.`date "+%Y%m%d_%H%M%S"`.bak
  ivector-subtract-global-mean scp:$nnet_dir/xvectors_$data_name/xvector.scp ark,scp:$nnet_dir/xvectors_$data_name/xvector.sub.ark,$nnet_dir/xvectors_$data_name/xvector.sub.scp || exit 1
  cp $nnet_dir/xvectors_$data_name/xvector.sub.scp $nnet_dir/xvectors_$data_name/xvector.scp
}

for d in $data;do
  substract_mean $d $nnet_dir
done
