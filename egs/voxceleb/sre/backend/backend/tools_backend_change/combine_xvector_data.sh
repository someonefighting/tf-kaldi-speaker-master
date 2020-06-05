#!/bin/bash

nnet_dir=$1
out_name=$2

[ -d $nnet_dir/xvectors_$out_name ] && mv $nnet_dir/xvectors_$out_name $nnet_dir/xvectors_$out_name.`date "+%Y%m%d_%H%M%S"`.bak
mkdir -p $nnet_dir/xvectors_$out_name

[ -d data/$out_name ] && mv data/$out_name data/$out_name.`date "+%Y%m%d_%H%M%S"`.bak
mkdir -p data/$out_name

combine_dir() {
  nnet_dir=$1
  out_name=$2
  in_name=$3
  cat $nnet_dir/xvectors_$in_name/xvector.scp >> $nnet_dir/xvectors_$out_name/xvector.scp
  cat $nnet_dir/xvectors_$in_name/spk_xvector.scp >> $nnet_dir/xvectors_$out_name/spk_xvector.scp
  cat $nnet_dir/xvectors_$in_name/num_utts.ark >> $nnet_dir/xvectors_$out_name/num_utts.ark
  cat data/$in_name/spk2utt >> data/$out_name/spk2utt
  cat data/$in_name/utt2spk >> data/$out_name/utt2spk
}


while true;do
  in_name=$3
  [ -z $in_name ] && exit 0
  shift
  combine_dir $nnet_dir $out_name $in_name
done

