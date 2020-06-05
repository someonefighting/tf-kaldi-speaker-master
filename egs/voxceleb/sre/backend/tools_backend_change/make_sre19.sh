#!/bin/bash

nnet_dir=$1

[ -d $nnet_dir/xvectors_sre19 ] && mv $nnet_dir/xvectors_sre19 $nnet_dir/xvectors_sre19.`date "+%Y%m%d_%H%M%S"`.bak
[ -d data/sre19 ] && mv data/sre19 data/sre19.`date "+%Y%m%d_%H%M%S"`.bak
mkdir $nnet_dir/xvectors_sre19

cat $nnet_dir/xvectors_sre19_eval_enroll/xvector.scp >> $nnet_dir/xvectors_sre19/xvector.scp
cat $nnet_dir/xvectors_sre19_eval_test/xvector.scp >> $nnet_dir/xvectors_sre19/xvector.scp

cat $nnet_dir/xvectors_sre19_eval_enroll/spk_xvector.scp >> $nnet_dir/xvectors_sre19/spk_xvector.scp
cat $nnet_dir/xvectors_sre19_eval_test/spk_xvector.scp >> $nnet_dir/xvectors_sre19/spk_xvector.scp

cat $nnet_dir/xvectors_sre19_eval_enroll/num_utts.ark >> $nnet_dir/xvectors_sre19/num_utts.ark
cat $nnet_dir/xvectors_sre19_eval_test/num_utts.ark >> $nnet_dir/xvectors_sre19/num_utts.ark

cp -r tools_backend_change/data/sre19 data/
