#!/bin/bash
# Copyright      2019   Tianyu Liang, Can Xu
# Apache 2.0.

## must run with python3
## you must already extract x-vectors before run this project!

. ./cmd.sh
. ./path.sh
set -e

root=`pwd`
data=$root/data
exp=$root/exp_eftdnn_am
nnet_dir=$exp/xvector_nnet_tdnn_amsoftmax_m0.20_linear_bn_1e-2

data_for_mean_vec="sre18_major"
data_for_lda="sre_combined_sre18"
data_for_whiten="$data_for_lda" #same as lda
data_for_plda="$data_for_lda" #same as lda
data_for_adapt="sre_combined_sre18"
data_for_asnorm="sre18_major" #must be the same signal path as the dev and eval dataset

## must be split by '\n' not 'space'
## do not leave blank lines
data_for_dev="sre18_dev
sre18_eval"
data_for_eval="sre19_eval"

## trails-files and key-files, note that dev dataset must have both trails-file and key-file, eval dataset must have trails-file
sre18_dev_trials_file=LDC2019E59/dev/docs/sre18_dev_trials.tsv
sre18_dev_trials_key=LDC2019E59/dev/docs/sre18_dev_trial_key.tsv
sre18_eval_trials_file=LDC2019E59/eval/docs/sre18_eval_trials.tsv
sre18_eval_trials_key=LDC2019E59/eval/docs/sre18_eval_trial_key.tsv
sre19_eval_trials_file=LDC2019E58/docs/sre19_cts_challenge_trials.tsv

lda_dim=150

nj=32

stage=1

for d in $data_for_dev;do
  [ -f ${d}.result ] && mv ${d}.result ${d}.`date "+%Y%m%d_%H%M%S"`.result
done

## clear all cache like lda, plda ...
if [ $stage -le -100 ]; then
  for f in `find $nnet_dir/ -name plda*` `find $nnet_dir/ -name transform.mat` `find $nnet_dir/ -name whiten.mat` `find $nnet_dir/ -name mean.vec` ;do
    rm $f
  done
  [ -d $nnet_dir/xvector_scores ] && mv $nnet_dir/xvector_scores $nnet_dir/xvector_scores.`date "+%Y%m%d_%H%M%S"`.bak
fi
## check if xvectors not exist
if [ $stage -le 1 ]; then
  for name in "`echo $data_for_mean_vec" "$data_for_lda" "$data_for_whiten" "$data_for_plda" "$data_for_adapt | sed -e 's/ /\n/g' | sort -u`" `echo "$data_for_dev" | sed -e 's/$/_enroll/g'` `echo "$data_for_dev" | sed -e 's/$/_test/g'` `echo "$data_for_eval" | sed -e 's/$/_enroll/g'` `echo "$data_for_eval" | sed -e 's/$/_test/g'`; do
    [ ! -f $nnet_dir/xvectors_$name/xvector.scp ] && echo "[ERROR] check xvectors : $name. File $nnet_dir/xvectors_$name/xvector.scp does not exist!" && exit 1
    echo "[INFO] check xvectors : $name. Check OK!"
  done
  echo "stage 1 finish!"
fi
## compute mean vector
if [ $stage -le 2 ]; then
  for name in $data_for_mean_vec;do
    [ -f $nnet_dir/xvectors_$name/mean.vec ] && echo "[INFO] ivector mean : skip $name. It has been processed fefore." && continue
    { $train_cmd $nnet_dir/xvectors_$name/log/compute_mean.log \
      ivector-mean scp:$nnet_dir/xvectors_$name/xvector.scp \
      $nnet_dir/xvectors_$name/mean.vec
    echo "[INFO] ivector mean : $name done!";} &
  done
  wait
  echo "stage 2 finish!"
fi
## compute lda
if [ $stage -le 3 ]; then
  for name in $data_for_lda;do
    [ -f $nnet_dir/xvectors_$name/transform.mat ] && echo "[INFO] compute lda : skip $name. It has been processed fefore." && continue
    { $train_cmd $nnet_dir/xvectors_$name/log/lda.log \
      ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
      "ark:ivector-subtract-global-mean scp:$nnet_dir/xvectors_$name/xvector.scp ark:- |" \
      ark:$name/utt2spk $nnet_dir/xvectors_$name/transform.mat
    echo "[INFO] compute lda : $name done!";} &
  done
  wait
  echo "stage 3 finish!"
fi
## compute lda-whiten
if [ $stage -le 4 ]; then
  for name in $data_for_whiten;do
    [ -f $nnet_dir/xvectors_$name/whiten.mat ] && echo "[INFO] whiten : skip $name. It has been processed fefore." && continue
    [ ! -f $nnet_dir/xvectors_$name/transform.mat ] && echo "[ERROR] whiten $name : $nnet_dir/xvectors_$name/transform.mat does not exists!" && continue
    { $train_cmd $nnet_dir/xvectors_$name/log/whiten.log \
      est-pca --read-vectors=true --normalize-mean=false --normalize-variance=true \
      "ark:ivector-subtract-global-mean scp:$nnet_dir/xvectors_$name/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_$name/transform.mat ark:- ark:- |" \
      $nnet_dir/xvectors_$name/whiten.mat
    echo "[INFO] whiten : $name done!";} &
  done
  wait
  echo "stage 4 finish!"
fi
## compute plda
if [ $stage -le 5 ]; then
  for name in $data_for_plda;do
    [ -f $nnet_dir/xvectors_$name/plda ] && echo "[INFO] compute plda : skip $name. It has been processed fefore." && continue
    [ ! -f $nnet_dir/xvectors_$name/transform.mat ] && echo "[ERROR] compute plda : $name $nnet_dir/xvectors_$name/transform.mat does not exists!" && exit 1
    { $train_cmd $nnet_dir/xvectors_$name/log/plda.log \
      ivector-compute-plda ark:$name/spk2utt \
      "ark:ivector-subtract-global-mean scp:$nnet_dir/xvectors_$name/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_$name/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
      $nnet_dir/xvectors_$name/plda
    echo "[INFO] compute plda : $name done!";} &
  done
  for name in $data_for_plda;do
    [ -f $nnet_dir/xvectors_$name/plda_whiten ] && echo "[INFO] compute whiten plda : skip $name. It has been processed fefore." && continue
    [ ! -f $nnet_dir/xvectors_$name/whiten.mat ] && echo "[ERROR] compute whiten plda : $name $nnet_dir/xvectors_$name/whiten.mat does not exists!" && exit 1
    { $train_cmd $nnet_dir/xvectors_$name/log/plda_whiten.log \
      ivector-compute-plda ark:$name/spk2utt \
      "ark:ivector-subtract-global-mean scp:$nnet_dir/xvectors_$name/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_$name/transform.mat ark:- ark:- | transform-vec $nnet_dir/xvectors_$name/whiten.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
      $nnet_dir/xvectors_$name/plda_whiten
    echo "[INFO] compute whiten plda : $name done!";} &
  done
  wait
  echo "stage 5 finish!"
fi
## adapt plda
if [ $stage -le 6 ]; then
  for plda in $data_for_plda;do
    for name in $data_for_adapt;do
      [ -f $nnet_dir/xvectors_$plda/plda_adapt_${name} ] && echo "[INFO] adapt plda : skip $plda with $name. It has been processed fefore." && continue
      [ ! -f $nnet_dir/xvectors_$plda/plda ] && echo "[ERROR] adapt plda : $plda with $name $nnet_dir/xvectors_$plda/plda does not exists!" && exit 1
      { $train_cmd $nnet_dir/xvectors_$plda/log/plda_adapt_${name}.log \
        ivector-adapt-plda --within-covar-scale=0.75 --between-covar-scale=0.25 \
        $nnet_dir/xvectors_$plda/plda \
        "ark:ivector-subtract-global-mean scp:$nnet_dir/xvectors_$name/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_$plda/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        $nnet_dir/xvectors_$plda/plda_adapt_${name}
      echo "[INFO] adapt plda : $plda with $name done!";} &
    done
    for name in $data_for_adapt;do
      [ -f $nnet_dir/xvectors_$plda/plda_whiten_adapt_${name} ] && echo "[INFO] adapt whiten plda : skip $plda with $name. It has been processed fefore." && continue
      [ ! -f $nnet_dir/xvectors_$plda/plda_whiten ] && echo "[ERROR] adapt whiten plda : $plda with $name $nnet_dir/xvectors_$plda/plda_whiten does not exists!" && exit 1
      { $train_cmd $nnet_dir/xvectors_$plda/log/plda_whiten_adapt_${name}.log \
        ivector-adapt-plda --within-covar-scale=0.75 --between-covar-scale=0.25 \
        $nnet_dir/xvectors_$plda/plda_whiten \
        "ark:ivector-subtract-global-mean scp:$nnet_dir/xvectors_$name/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_$plda/transform.mat ark:- ark:- | transform-vec $nnet_dir/xvectors_$plda/whiten.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        $nnet_dir/xvectors_$plda/plda_whiten_adapt_${name}
      echo "[INFO] adapt whiten plda : $plda with $name done!";} &
    done
  done
  wait
  echo "stage 6 finish!"
fi
## function compute score without asnorm
compute_score() {
  test_data=$1
  mean_vec_data=$2
  plda_data=$3
  if_whiten=$4
  adapt_data=$5
  asnorm_data=$6

  mean_vec_file=$nnet_dir/xvectors_$mean_vec_data/mean.vec
  transform_mat=$nnet_dir/xvectors_$plda_data/transform.mat
  whiten_mat=$nnet_dir/xvectors_$plda_data/whiten.mat
  plda_model=$nnet_dir/xvectors_$plda_data/plda
  text_whiten=""
  if [ $if_whiten = yes ];then
    plda_model=${plda_model}_whiten
    text_whiten=" | transform-vec $whiten_mat ark:- ark:-"
  fi
  if [ $adapt_data != no ];then
    plda_model=${plda_model}_adapt_${adapt_data}
  fi

  text_prefix=${test_data}_mean_${mean_vec_data}_plda_${plda_data}_whiten_${if_whiten}_adapt_${adapt_data}_asnorm_${asnorm_data}

  if [ $asnorm_data = no ];then
    #echo "[INFO] conmuting scores on $test_data"

    $train_cmd $nnet_dir/xvector_scores/log/${text_prefix}_scores.log \
      ivector-plda-scoring --normalize-length=true \
      --num-utts=ark:$nnet_dir/xvectors_${test_data}_enroll/num_utts.ark \
      "ivector-copy-plda --smoothing=0.0 $plda_model - |" \
      "ark:ivector-mean ark:$data/${test_data}_enroll/spk2utt scp:$nnet_dir/xvectors_${test_data}_enroll/xvector.scp ark:- | ivector-subtract-global-mean $mean_vec_file ark:- ark:- | transform-vec $transform_mat ark:- ark:-${text_whiten} | ivector-normalize-length ark:- ark:- |" \
      "ark:ivector-subtract-global-mean $mean_vec_file scp:$nnet_dir/xvectors_${test_data}_test/xvector.scp ark:- | transform-vec $transform_mat ark:- ark:-${text_whiten} | ivector-normalize-length ark:- ark:- |" \
      "cat '$data/${test_data}_test/trials' | cut -d\  --fields=1,2 |" $nnet_dir/xvector_scores/${text_prefix}_scores ;
    bash tools/kaldiscore2nistscore.sh $nnet_dir/xvector_scores/${text_prefix}_scores $nnet_dir/xvector_scores/${text_prefix}_scores.tsv
  else
    #echo "[INFO] conmuting asnorm scores on $test_data"

    text_enroll_prefix=${test_data}_enroll_mean_${mean_vec_data}_plda_${plda_data}_whiten_${if_whiten}_adapt_${adapt_data}_asnorm_${asnorm_data}
    text_test_prefix=${test_data}_test_mean_${mean_vec_data}_plda_${plda_data}_whiten_${if_whiten}_adapt_${adapt_data}_asnorm_${asnorm_data}

    enroll_spk_list=$data/${test_data}_enroll/spk
    test_spk_list=$data/${test_data}_test/spk
    cut -d" " $data/${test_data}_enroll/spk2utt -f1 > ${enroll_spk_list}
    cut -d" " $data/${test_data}_test/spk2utt -f1 > ${test_spk_list}
    mean_vec_spk_list=data/${mean_vec_data}/spk
    cut -d" " data/${mean_vec_data}/spk2utt -f1 > ${mean_vec_spk_list}

    trials_file=`eval echo "$"${test_data}_trials_file`

    $train_cmd $nnet_dir/xvector_scores/log/${text_prefix}_scores_matrix.log \
      tools/ivector-plda-scoring-matrix --normalize-length=true \
      --num-utts=ark:$nnet_dir/xvectors_${test_data}_enroll/num_utts.ark \
      "ivector-copy-plda --smoothing=0.0 ${plda_model} - |" \
      "ark:ivector-mean ark:$data/${test_data}_enroll/spk2utt scp:$nnet_dir/xvectors_${test_data}_enroll/xvector.scp ark:- | ivector-subtract-global-mean ${mean_vec_file} ark:- ark:- | transform-vec ${transform_mat} ark:- ark:-${text_whiten} | ivector-normalize-length ark:- ark:- |" \
      "ark:ivector-subtract-global-mean ${mean_vec_file} scp:$nnet_dir/xvectors_${test_data}_test/xvector.scp ark:- | transform-vec ${transform_mat} ark:- ark:-${text_whiten} | ivector-normalize-length ark:- ark:- |" \
      $enroll_spk_list $test_spk_list $nnet_dir/xvector_scores/${text_prefix}_scores_matrix ;

    $train_cmd $nnet_dir/xvector_scores/log/${text_enroll_prefix}_scores_matrix_snorm.log \
      tools/ivector-plda-scoring-matrix --normalize-length=true \
      --num-utts=ark:$nnet_dir/xvectors_${test_data}_enroll/num_utts.ark \
      "ivector-copy-plda --smoothing=0.0 ${plda_model} - |" \
      "ark:ivector-mean ark:$data/${test_data}_enroll/spk2utt scp:$nnet_dir/xvectors_${test_data}_enroll/xvector.scp ark:- | ivector-subtract-global-mean ${mean_vec_file} ark:- ark:- | transform-vec ${transform_mat} ark:- ark:-${text_whiten} | ivector-normalize-length ark:- ark:- |" \
      "ark:ivector-subtract-global-mean ${mean_vec_file} scp:$nnet_dir/xvectors_${asnorm_data}/xvector.scp ark:- | transform-vec ${transform_mat} ark:- ark:-${text_whiten} | ivector-normalize-length ark:- ark:- |" \
      ${enroll_spk_list} ${mean_vec_spk_list} $nnet_dir/xvector_scores/${text_enroll_prefix}_scores_matrix_majorsnorm ;

    $train_cmd $nnet_dir/xvector_scores/log/${text_test_prefix}_scores_matrix_snorm.log \
      tools/ivector-plda-scoring-matrix --normalize-length=true \
      --num-utts=ark:$nnet_dir/xvectors_${test_data}_test/num_utts.ark \
      "ivector-copy-plda --smoothing=0.0 ${plda_model} - |" \
      "ark:ivector-mean ark:$data/${test_data}_test/spk2utt scp:$nnet_dir/xvectors_${test_data}_test/xvector.scp ark:- | ivector-subtract-global-mean ${mean_vec_file} ark:- ark:- | transform-vec ${transform_mat} ark:- ark:-${text_whiten} | ivector-normalize-length ark:- ark:- |" \
      "ark:ivector-subtract-global-mean ${mean_vec_file} scp:$nnet_dir/xvectors_${asnorm_data}/xvector.scp ark:- | transform-vec ${transform_mat} ark:- ark:-${text_whiten} | ivector-normalize-length ark:- ark:- |" \
      ${test_spk_list} ${mean_vec_spk_list} $nnet_dir/xvector_scores/${text_test_prefix}_scores_matrix_majorsnorm ;

    python tools/s-norm-get-enroll.py --score-file $nnet_dir/xvector_scores/${text_enroll_prefix}_scores_matrix_majorsnorm --enroll-file $nnet_dir/xvector_scores/${text_enroll_prefix}.snorm 1>/dev/null

    python tools/s-norm-get-enroll.py --score-file $nnet_dir/xvector_scores/${text_test_prefix}_scores_matrix_majorsnorm --enroll-file $nnet_dir/xvector_scores/${text_test_prefix}.snorm 1>/dev/null

    python tools/s-norm-from-enroll.py --score-file $nnet_dir/xvector_scores/${text_prefix}_scores_matrix --snorm-file $nnet_dir/xvector_scores/${text_prefix}_scores_matrix_snorm --enroll-file $nnet_dir/xvector_scores/${text_enroll_prefix}.snorm --eval-file $nnet_dir/xvector_scores/${text_test_prefix}.snorm 1>/dev/null

    python tools/matrix-kaldi.py --matrix-file $nnet_dir/xvector_scores/${text_prefix}_scores_matrix_snorm --trial-file ${trials_file} --kaldi-file $nnet_dir/xvector_scores/${text_prefix}_scores.tsv 1>/dev/null
  fi
}
## function compute eer with score file
compute_result() {
  test_data=$1
  mean_vec_data=$2
  plda_data=$3
  if_whiten=$4
  adapt_data=$5
  asnorm_data=$6

  #echo "[INFO] conmuting result on $test_data"

  text_prefix=${test_data}_mean_${mean_vec_data}_plda_${plda_data}_whiten_${if_whiten}_adapt_${adapt_data}_asnorm_${asnorm_data}
  score_file=$nnet_dir/xvector_scores/${text_prefix}_scores.tsv

  trials_file=`eval echo "$"${test_data}_trials_file`
  trials_key=`eval echo "$"${test_data}_trials_key`

  score_tmp=$(python tools/cts_challenge_scoring_software/sre18_submission_scorer.py -o ${score_file} -l ${trials_file} -r ${trials_key} 2>/dev/null) && score_tmp=`echo "$score_tmp" | sed -n '3p' | awk '{print $2" "$3" "$4}'`
  echo "${test_data} result : ${score_tmp}
mean : $mean_vec_data
lda : $plda_data
whiten : ${if_whiten}
plda : $plda_data
adapt : $adapt_data
asnorm : $asnorm_data
score file : ${score_file}
"
  echo -e "${mean_vec_data}\t${plda_data}\t${if_whiten}\tyes\t${adapt_data}\t${asnorm_data}\t${score_tmp}" >> ${test_data}.result
}
## loop for compute
if [ $stage -le 7 ]; then
  fifo_thread_num=6
  tmp_fifofile="/tmp/$.fifo"
  [ -f $tmp_fifofile ] && rm $tmp_fifofile
  mkfifo $tmp_fifofile
  eval "exec ${fifo_thread_num}<>$tmp_fifofile"
  rm $tmp_fifofile
  for ((i=0;i<$nj;i++));do echo ; done >&$fifo_thread_num

  num_total=$((`echo "$data_for_mean_vec" |wc -l`*`echo "$data_for_plda" |wc -l`*2*(1+`echo "$data_for_adapt" |wc -l`)*(`echo "$data_for_asnorm" |wc -l`+1)*(`echo "$data_for_dev" |wc -l`+`echo "$data_for_eval" |wc -l`)))
  num_finished=0
  fifo_progress_num=7
  tmp_fifofile="/tmp/$.fifo"
  [ -f $tmp_fifofile ] && rm $tmp_fifofile
  mkfifo $tmp_fifofile
  eval "exec ${fifo_progress_num}<>$tmp_fifofile"
  rm $tmp_fifofile
  printf "progress : %d/%d done!\r" $num_finished $num_total
  echo $num_finished >&$fifo_progress_num

  for mean_vec_data in $data_for_mean_vec;do
  for plda_data in $data_for_plda;do
  for if_whiten in yes no;do
  for adapt_data in no $data_for_adapt;do
  for asnorm_data in $data_for_asnorm no;do
  for test_data in $data_for_dev;do
    read -u$fifo_thread_num
    { [ ! -f $nnet_dir/xvector_scores/${test_data}_mean_${mean_vec_data}_plda_${plda_data}_whiten_${if_whiten}_adapt_${adapt_data}_asnorm_${asnorm_data}_scores.tsv ] && compute_score $test_data $mean_vec_data $plda_data $if_whiten $adapt_data $asnorm_data;
    compute_result $test_data $mean_vec_data $plda_data $if_whiten $adapt_data $asnorm_data;
    read -u$fifo_progress_num num_finished;
    printf "progress : %d/%d done!\r" $((num_finished+1)) $num_total
    echo $((num_finished+1)) >&$fifo_progress_num;
    echo >&$fifo_thread_num;} &
  done
  for test_data in $data_for_eval;do
    read -u$fifo_thread_num
    { [ ! -f $nnet_dir/xvector_scores/${test_data}_mean_${mean_vec_data}_plda_${plda_data}_whiten_${if_whiten}_adapt_${adapt_data}_asnorm_${asnorm_data}_scores.tsv ] && compute_score $test_data $mean_vec_data $plda_data $if_whiten $adapt_data $asnorm_data;
    read -u$fifo_progress_num num_finished;
    printf "progress : %d/%d done!\r" $((num_finished+1)) $num_total
    echo $((num_finished+1)) >&$fifo_progress_num;
    echo >&$fifo_thread_num;} &
  done
  done
  done
  done
  done
  done

  wait
  eval "exec ${fifo_thread_num}>&-"
  eval "exec ${fifo_progress_num}>&-"
  echo "stage 7 finish!"
fi

echo "finish!"
