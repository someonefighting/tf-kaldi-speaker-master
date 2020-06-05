#!/bin/bash

nnet_dir=exp/adsf

. path.sh

## change lda_dim to 200
sed -i 's#^lda_dim=.*$#lda_dim=200#g' run_after_extract_xvector.sh

## change asnorm top n to 0.3
sed -i 's#score_ww = score_ww\[-int(leng.*):\]#score_ww = score_ww\[-int(leng*0.3):\]#g' tools/s-norm-get-enroll.py

## prepare sre18_major list
#bash tools_backend_change/make_sre18_major_combined_list.sh $nnet_dir

## xvectors substract own mean
bash tools_backend_change/xvector_substract_own_mean_sre_sre16_sre18_combined.sh $nnet_dir || exit 1
bash tools_backend_change/xvector_substract_own_mean_sre16_sre18_combined.sh $nnet_dir || exit 1
for d in sre18_major_combined call_home_friend_combined sre18_dev_enroll sre18_dev_test sre18_eval_enroll sre18_eval_test sre19_eval_enroll sre19_eval_test;do
  bash tools_backend_change/xvector_substract_own_mean.sh $nnet_dir $d || exit 1
done

bash tools_backend_change/make_sre19.sh $nnet_dir || exit 1

## make xvectors combined directory
bash tools_backend_change/combine_xvector_data.sh $nnet_dir sre_sre16_sre18_call_home_friend_sre18_major_combined_sre19 sre_sre16_sre18_combined call_home_friend_combined sre18_major_combined sre19

bash tools_backend_change/combine_xvector_data.sh $nnet_dir sre16_sre18_call_home_friend_sre18_major_combined_sre19 sre16_sre18_combined call_home_friend_combined sre18_major_combined sre19

bash tools_backend_change/combine_xvector_data.sh $nnet_dir sre_sre16_sre18_combined_sre19 sre_sre16_sre18_combined sre19
