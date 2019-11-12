#!/bin/bash
# Copyright      2019   Can Xu
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (mostly EERs) are inline in comments below.
#
# This example demonstrates a "bare bones" NIST SRE 2019 recipe using xvectors.
# It is closely based on "X-vectors: Robust DNN Embeddings for Speaker
# Recognition" by Snyder et al.  In the future, we will add score-normalization
# and a more effective form of PLDA domain adaptation.
#

#此脚本仅用于整理训练数据使用(stage=0,步骤0)，步骤1之后内容仅供参考, 请根据各自系统安排步骤1（包含）之后的操作。GOOD LUCK

. ./cmd.sh
. ./path.sh
set -e
kaldi_voxceleb=/liry_tf/kaldi/egs/voxceleb
root=/liry_tf/tf-kaldi-speaker/egs/voxceleb/sre19_demo

mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

sre18_dev_trials=data/sre18_dev_test/trials
sre18_eval_trials=data/sre18_eval_test/trials
sre19_eval_trials=data/sre19_eval_test/trials

nnet_dir=exp/xvector_nnet_1a

voxceleb1_root=/liry_tf/Data/voxceleb1
voxceleb2_root=/liry_tf/Data/voxceleb2

data_root=/liry_tf/Data/SRE18/
sre19_root=/liry_tf/Data/SRE19/

stage=3
train_stage=-1
nj=90

if [ $stage -le -1 ]; then
    # link the directories
    rm -fr utils steps sid
    ln -s $kaldi_voxceleb/v2/utils ./
    ln -s $kaldi_voxceleb/v2/steps ./
    ln -s $kaldi_voxceleb/v2/sid ./
#    ln -s $kaldi_voxceleb/v2/conf ./
#    ln -s $kaldi_voxceleb/v2/local ./
fi

if [ $stage -le 0 ]; then
  # Prepare telephone and microphone speech from Mixer6.
  local/make_mx6.sh ${data_root}/ data/

  # Prepare SRE18 test and enroll.
  local/make_sre18.pl ${sre19_root}/LDC2019E59/eval data/

  # Prepare SRE16 test and enroll.
  local/make_sre16.pl ${data_root}/SRE16/data/eval/R149_0_1/ data

  # Prepare SRE10 test and enroll. Includes microphone interview speech.
  # NOTE: This corpus is now available through the LDC as LDC2017S06.
  local/make_sre10.pl ${data_root}/SRE10/eval/ data/

  # Prepare SRE08 test and enroll. Includes some microphone speech.
  local/make_sre08.pl $data_root/SRE08/sp08-11/test/ $data_root/SRE08/sp08-11/train/ data/

  # This prepares the older NIST SREs from 2004-2006.
  local/make_sre.sh ${data_root} data/

  # Combine all SREs prior to 2016 and Mixer6 into one dataset
  utils/combine_data.sh data/sre \
    data/sre2004 data/sre2005_train \
    data/sre2005_test data/sre2006_train \
    data/sre2006_test data/sre08 data/mx6 data/sre10 data/sre16
  utils/validate_data_dir.sh --no-text --no-feats data/sre
  utils/fix_data_dir.sh data/sre

  # Prepare SWBD corpora.
  local/make_swbd_cellular1.pl ${data_root}/LDC2018E48_Comprehensive_Switchboard/LDC2001S13 \
    data/swbd_cellular1_train
  local/make_swbd_cellular2.pl ${data_root}/LDC2018E48_Comprehensive_Switchboard/LDC2004S07 \
    data/swbd_cellular2_train
  local/make_swbd2_phase1.pl ${data_root}/LDC2018E48_Comprehensive_Switchboard/LDC98S75 \
    data/swbd2_phase1_train
  local/make_swbd2_phase2.pl ${data_root}/LDC2018E48_Comprehensive_Switchboard/LDC99S79 \
    data/swbd2_phase2_train
  local/make_swbd2_phase3.pl ${data_root}/LDC2018E48_Comprehensive_Switchboard/LDC2002S06 \
    data/swbd2_phase3_train

  # Combine all SWB corpora into one dataset.
  utils/combine_data.sh data/swbd \
    data/swbd_cellular1_train data/swbd_cellular2_train \
    data/swbd2_phase1_train data/swbd2_phase2_train data/swbd2_phase3_train

  # Default 8k, if 16k is required, go to local/make_voxceleb2.pl:53
  local/make_voxceleb2.pl ${voxceleb2_root} dev data/voxceleb2_train
  local/make_voxceleb2.pl ${voxceleb2_root} test data/voxceleb2_test
  # This script creates data/voxceleb1_test and data/voxceleb1_train for latest version of VoxCeleb1.
  # Our evaluation set is the test portion of VoxCeleb1.
  # Default 8k, if 16k is required, go to local/make_voxceleb1.pl:83
  local/make_voxceleb1.pl $voxceleb1_root data
  # We'll train on all of VoxCeleb2, plus the training portion of VoxCeleb1.
  # This should give 7,323 speakers and 1,276,888 utterances.
  utils/combine_data.sh data/voxceleb data/voxceleb2_train data/voxceleb2_test data/voxceleb1_train data/voxceleb1_test

  #Prepare NIST SRE 2018 DEV data
  local/make_sre18_dev.pl ${sre19_root}/LDC2019E59/dev/ data

  #Prepare NIST SRE 2018 EVAL data
  local/make_sre18_eval.pl ${sre19_root}/LDC2019E59/eval/ data

  #Prepare NIST SRE 2018 UNLABEL data
  local/make_sre18_unlabeled.pl ${sre19_root}/LDC2019E59/dev/ data

  #Prepare NIST SRE 2019 EVAL data
  local/make_sre19_eval.pl ${sre19_root}/LDC2019E58/ data

  utils/combine_data.sh data/sre16_sre18 data/sre16 data/sre18

fi

sre18_major_list=data/sre18_major/spk
cut -d" " data/sre18_major/spk2utt -f1 > ${sre18_major_list}

#以下内容仅供参考，并未验证

if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in sre swbd voxceleb sre18_dev_enroll sre18_dev_test \
              sre18_eval_enroll sre18_eval_test sre18_major       \
              sre19_eval_enroll sre19_eval_test sre16_sre18; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 90 --cmd "$train_cmd" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 90 --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
  done
fi

# In this section, we augment the SWBD and SRE data with reverberation,
# noise, music, and babble, and combined it with the clean data.
# The combined list will be used to train the xvector DNN.  The SRE
# subset will be used to train the PLDA model.
if [ $stage -le 2 ]; then
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/sre16_sre18/utt2num_frames > data/sre16_sre18/reco2dur
  if [ ! -d "RIRS_NOISES" ]; then
    if [ ! -d "${data_root}/../RIRS_NOISES" ]; then
      # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
      wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
      unzip rirs_noises.zip
    else
      ln -s ${data_root}/../RIRS_NOISES ./
    fi
  fi

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the SRE16+SRE18 list.  Note that we don't add any
  # additive noise here.
  python steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 8000 \
    data/sre16_sre18 data/sre16_sre18_reverb
  cp data/sre16_sre18/vad.scp data/sre16_sre18_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" data/sre16_sre18_reverb data/sre16_sre18_reverb.new
  rm -rf data/sre16_sre18_reverb
  mv data/sre16_sre18_reverb.new data/sre16_sre18_reverb

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  steps/data/make_musan.sh --sampling-rate 8000 ${data_root}/../musan data

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh data/musan_${name}
    mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
  done

  # Augment with musan_noise
  python steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/sre16_sre18 data/sre16_sre18_noise
  # Augment with musan_music
  python steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/sre16_sre18 data/sre16_sre18_music
  # Augment with musan_speech
  python steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/sre16_sre18 data/sre16_sre18_babble

  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh data/sre16_sre18_aug data/sre16_sre18_reverb data/sre16_sre18_noise data/sre16_sre18_music data/sre16_sre18_babble

  # Take a random subset of the augmentations (128k is somewhat larger than twice
  # the size of the SWBD+SRE list)
  num=`wc -l data/sre16_sre18/utt2spk | cut -d" " -f1`
  utils/subset_data_dir.sh data/sre16_sre18_aug ${num} data/sre16_sre18_aug_${num}
  utils/fix_data_dir.sh data/sre16_sre18_aug_${num}

  # Make MFCCs for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj $nj --cmd "$train_cmd" \
    data/sre16_sre18_aug_${num} exp/make_mfcc $mfccdir

  # Combine the clean and augmented SWBD+SRE list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh data/sre16_sre18_combined data/sre16_sre18_aug_${num} data/sre16_sre18

  # Filter out the clean + augmented portion of the SRE list.  This will be used to
  # train the PLDA model later in the script.
  utils/fix_data_dir.sh data/sre16_sre18_combined
fi

# In this section, we augment the SWBD and SRE data with reverberation,
# noise, music, and babble, and combined it with the clean data.
# The combined list will be used to train the xvector DNN.  The SRE
# subset will be used to train the PLDA model.
if [ $stage -le 3 ]; then
  utils/combine_data.sh --extra-files 'utt2num_frames' data/swbd_sre_voxceleb data/sre data/swbd data/voxceleb
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/swbd_sre_voxceleb/utt2num_frames > data/swbd_sre_voxceleb/reco2dur

  # create soft-link of RIRS_NOISES resources in current directory
  if [ ! -d "RIRS_NOISES" ]; then
    if [ ! -d "${data_root}/../RIRS_NOISES" ]; then
      # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
      wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
      unzip rirs_noises.zip
    else
      ln -s ${data_root}/../RIRS_NOISES ./
    fi
  fi

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the SWBD+SRE list.  Note that we don't add any
  # additive noise here.
  python steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 8000 \
    data/swbd_sre_voxceleb data/swbd_sre_voxceleb_reverb
  cp data/swbd_sre_voxceleb/vad.scp data/swbd_sre_voxceleb_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" data/swbd_sre_voxceleb_reverb data/swbd_sre_voxceleb_reverb.new
  rm -rf data/swbd_sre_voxceleb_reverb
  mv data/swbd_sre_voxceleb_reverb.new data/swbd_sre_voxceleb_reverb

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  steps/data/make_musan.sh --sampling-rate 8000 ${data_root}/../musan data

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh data/musan_${name}
    mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
  done

  # Augment with musan_noise
  python steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/swbd_sre_voxceleb data/swbd_sre_voxceleb_noise
  # Augment with musan_music
  python steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/swbd_sre_voxceleb data/swbd_sre_voxceleb_music
  # Augment with musan_speech
  python steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/swbd_sre_voxceleb data/swbd_sre_voxceleb_babble

  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh data/swbd_sre_voxceleb_aug data/swbd_sre_voxceleb_reverb data/swbd_sre_voxceleb_noise data/swbd_sre_voxceleb_music data/swbd_sre_voxceleb_babble

  # Take a random subset of the augmentations (128k is somewhat larger than twice
  # the size of the SWBD+SRE list)
  #utils/subset_data_dir.sh data/swbd_sre_voxceleb_aug 1372000 data/swbd_sre_voxceleb_aug_1372k
  #utils/fix_data_dir.sh data/swbd_sre_voxceleb_aug_1372k
  # Make MFCCs for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj $nj --cmd "$train_cmd" \
    data/swbd_sre_voxceleb_aug exp/make_mfcc $mfccdir

  # Combine the clean and augmented SWBD+SRE list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh data/swbd_sre_voxceleb_combined data/swbd_sre_voxceleb_aug data/swbd_sre_voxceleb

  # Filter out the clean + augmented portion of the SRE list.  This will be used to
  # train the PLDA model later in the script.
  utils/copy_data_dir.sh data/swbd_sre_voxceleb_combined data/sre_combined
  utils/filter_scp.pl data/sre/spk2utt data/swbd_sre_voxceleb_combined/spk2utt | utils/spk2utt_to_utt2spk.pl > data/sre_combined/utt2spk
  utils/fix_data_dir.sh data/sre_combined
fi

  # Now we prepare the features to generate examples for xvector training.
if [ $stage -le 4 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj $nj --cmd "$train_cmd" \
    data/swbd_sre_voxceleb_combined data/swbd_sre_voxceleb_combined_no_sil exp/swbd_sre_voxceleb_combined_no_sil
  utils/fix_data_dir.sh data/swbd_sre_voxceleb_combined_no_sil
fi

if [ $stage -le 5  ]; then
  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 5s (500 frames) per utterance.
  min_len=400
  mv data/swbd_sre_voxceleb_combined_no_sil/utt2num_frames data/swbd_sre_voxceleb_combined_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/swbd_sre_voxceleb_combined_no_sil/utt2num_frames.bak > data/swbd_sre_voxceleb_combined_no_sil/utt2num_frames
  utils/filter_scp.pl data/swbd_sre_voxceleb_combined_no_sil/utt2num_frames data/swbd_sre_voxceleb_combined_no_sil/utt2spk > data/swbd_sre_voxceleb_combined_no_sil/utt2spk.new
  mv data/swbd_sre_voxceleb_combined_no_sil/utt2spk.new data/swbd_sre_voxceleb_combined_no_sil/utt2spk
  utils/fix_data_dir.sh data/swbd_sre_voxceleb_combined_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=8
  awk '{print $1, NF-1}' data/swbd_sre_voxceleb_combined_no_sil/spk2utt > data/swbd_sre_voxceleb_combined_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/swbd_sre_voxceleb_combined_no_sil/spk2num | utils/filter_scp.pl - data/swbd_sre_voxceleb_combined_no_sil/spk2utt > data/swbd_sre_voxceleb_combined_no_sil/spk2utt.new
  mv data/swbd_sre_voxceleb_combined_no_sil/spk2utt.new data/swbd_sre_voxceleb_combined_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl data/swbd_sre_voxceleb_combined_no_sil/spk2utt > data/swbd_sre_voxceleb_combined_no_sil/utt2spk

  utils/filter_scp.pl data/swbd_sre_voxceleb_combined_no_sil/utt2spk data/swbd_sre_voxceleb_combined_no_sil/utt2num_frames > data/swbd_sre_voxceleb_combined_no_sil/utt2num_frames.new
  mv data/swbd_sre_voxceleb_combined_no_sil/utt2num_frames.new data/swbd_sre_voxceleb_combined_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh data/swbd_sre_voxceleb_combined_no_sil
fi

if [ $stage -le 6 ]; then
  # Split the validation set
  num_heldout_spks=64
  num_heldout_utts_per_spk=20
  mkdir -p $data/voxceleb_train_combined_no_sil/train2/ $data/voxceleb_train_combined_no_sil/valid2/

  sed 's/-noise//' $data/voxceleb_train_combined_no_sil/utt2spk | sed 's/-music//' | sed 's/-babble//' | sed 's/-reverb//' |\
    paste -d ' ' $data/voxceleb_train_combined_no_sil/utt2spk - | cut -d ' ' -f 1,3 > $data/voxceleb_train_combined_no_sil/utt2uniq

  utils/utt2spk_to_spk2utt.pl $data/voxceleb_train_combined_no_sil/utt2uniq > $data/voxceleb_train_combined_no_sil/uniq2utt
  cat $data/voxceleb_train_combined_no_sil/utt2spk | utils/apply_map.pl -f 1 $data/voxceleb_train_combined_no_sil/utt2uniq |\
    sort | uniq > $data/voxceleb_train_combined_no_sil/utt2spk.uniq

  utils/utt2spk_to_spk2utt.pl $data/voxceleb_train_combined_no_sil/utt2spk.uniq > $data/voxceleb_train_combined_no_sil/spk2utt.uniq
  python $TF_KALDI_ROOT/misc/tools/sample_validset_spk2utt.py $num_heldout_spks $num_heldout_utts_per_spk $data/voxceleb_train_combined_no_sil/spk2utt.uniq > $data/voxceleb_train_combined_no_sil/valid2/spk2utt.uniq

  cat $data/voxceleb_train_combined_no_sil/valid2/spk2utt.uniq | utils/apply_map.pl -f 2- $data/voxceleb_train_combined_no_sil/uniq2utt > $data/voxceleb_train_combined_no_sil/valid2/spk2utt
  utils/spk2utt_to_utt2spk.pl $data/voxceleb_train_combined_no_sil/valid2/spk2utt > $data/voxceleb_train_combined_no_sil/valid2/utt2spk
  cp $data/voxceleb_train_combined_no_sil/feats.scp $data/voxceleb_train_combined_no_sil/valid2
  utils/filter_scp.pl $data/voxceleb_train_combined_no_sil/valid2/utt2spk $data/voxceleb_train_combined_no_sil/utt2num_frames > $data/voxceleb_train_combined_no_sil/valid2/utt2num_frames
  utils/fix_data_dir.sh $data/voxceleb_train_combined_no_sil/valid2

  utils/filter_scp.pl --exclude $data/voxceleb_train_combined_no_sil/valid2/utt2spk $data/voxceleb_train_combined_no_sil/utt2spk  > $data/voxceleb_train_combined_no_sil/train2/utt2spk
  utils/utt2spk_to_spk2utt.pl $data/voxceleb_train_combined_no_sil/train2/utt2spk > $data/voxceleb_train_combined_no_sil/train2/spk2utt
  cp $data/voxceleb_train_combined_no_sil/feats.scp $data/voxceleb_train_combined_no_sil/train2
  utils/filter_scp.pl $data/voxceleb_train_combined_no_sil/train2/utt2spk $data/voxceleb_train_combined_no_sil/utt2num_frames > $data/voxceleb_train_combined_no_sil/train2/utt2num_frames
  utils/fix_data_dir.sh $data/voxceleb_train_combined_no_sil/train2

  awk -v id=0 '{print $1, id++}' $data/voxceleb_train_combined_no_sil/train2/spk2utt > $data/voxceleb_train_combined_no_sil/train2/spklist
fi

if [ $stage -le 7 ]; then
## Training a softmax network
#nnetdir=$exp/xvector_nnet_tdnn_softmax_1e-2
#nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_softmax_1e-2.json \
#    $data/voxceleb_train_combined_no_sil/train $data/voxceleb_train_combined_no_sil/train/spklist \
#    $data/voxceleb_train_combined_no_sil/softmax_valid $data/voxceleb_train_combined_no_sil/train/spklist \
#    $nnetdir
#
#
## ASoftmax
#nnetdir=$exp/xvector_nnet_tdnn_asoftmax_m1_linear_bn_1e-2
#nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_asoftmax_m1_linear_bn_1e-2.json \
#    $data/voxceleb_train_combined_no_sil/train $data/voxceleb_train_combined_no_sil/train/spklist \
#    $data/voxceleb_train_combined_no_sil/softmax_valid $data/voxceleb_train_combined_no_sil/train/spklist \
#    $nnetdir
#
#nnetdir=$exp/xvector_nnet_tdnn_asoftmax_m2_linear_bn_1e-2
#nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_asoftmax_m2_linear_bn_1e-2.json \
#    $data/voxceleb_train_combined_no_sil/train $data/voxceleb_train_combined_no_sil/train/spklist \
#    $data/voxceleb_train_combined_no_sil/softmax_valid $data/voxceleb_train_combined_no_sil/train/spklist \
#    $nnetdir
#
# nnetdir=$exp/xvector_nnet_tdnn_asoftmax_m4_linear_bn_1e-2
# nnet/run_train_nnet_ftdnn_mgpu.sh --cmd "$cuda_cmd" --env tf_2_gpu --continue-training true nnet_conf/ftdnn2_asoftmax_m4_linear_bn_1e-2.json \
#     $data/voxceleb_train_combined_no_sil/train2 $data/voxceleb_train_combined_no_sil/train2/spklist \
#     $data/voxceleb_train_combined_no_sil/valid2 $data/voxceleb_train_combined_no_sil/train2/spklist \
#     $nnetdir
#
#
## Additive margin softmax
nnetdir=$exp/xvector_nnet_tdnn_amsoftmax_m0.15_linear_bn_1e-2
nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_amsoftmax_m0.15_linear_bn_1e-2.json \
    $data/voxceleb_train_combined_no_sil/train $data/voxceleb_train_combined_no_sil/train/spklist \
    $data/voxceleb_train_combined_no_sil/softmax_valid $data/voxceleb_train_combined_no_sil/train/spklist \
    $nnetdir
#
#nnetdir=$exp/xvector_nnet_tdnn_asoftmax_m4_linear_bn_1e-2
#nnet/run_train_nnet_eftdnn.sh --cmd "$cuda_cmd" --env tf_2_gpu --continue-training false nnet_conf/ftdnn2_asoftmax_m4_linear_bn_1e-2.json \
#    $data/voxceleb_train_combined_no_sil/train2 $data/voxceleb_train_combined_no_sil/train2/spklist \
#    $data/voxceleb_train_combined_no_sil/valid2 $data/voxceleb_train_combined_no_sil/train2/spklist \
#    $nnetdir
#
#nnetdir=$exp/xvector_nnet_tdnn_amsoftmax_m0.25_linear_bn_1e-2
#nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_amsoftmax_m0.25_linear_bn_1e-2.json \
#    $data/voxceleb_train_combined_no_sil/train $data/voxceleb_train_combined_no_sil/train/spklist \
#    $data/voxceleb_train_combined_no_sil/softmax_valid $data/voxceleb_train_combined_no_sil/train/spklist \
#    $nnetdir
#
#nnetdir=$exp/xvector_nnet_tdnn_amsoftmax_m0.30_linear_bn_1e-2
#nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_amsoftmax_m0.30_linear_bn_1e-2.json \
#    $data/voxceleb_train_combined_no_sil/train $data/voxceleb_train_combined_no_sil/train/spklist \
#    $data/voxceleb_train_combined_no_sil/softmax_valid $data/voxceleb_train_combined_no_sil/train/spklist \
#    $nnetdir
#
#nnetdir=$exp/xvector_nnet_tdnn_amsoftmax_m0.35_linear_bn_1e-2
#nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_amsoftmax_m0.35_linear_bn_1e-2.json \
#    $data/voxceleb_train_combined_no_sil/train $data/voxceleb_train_combined_no_sil/train/spklist \
#    $data/voxceleb_train_combined_no_sil/softmax_valid $data/voxceleb_train_combined_no_sil/train/spklist \
#    $nnetdir
#
## ArcSoftmax
#nnetdir=$exp/xvector_nnet_tdnn_arcsoftmax_m0.15_linear_bn_1e-2
#nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_arcsoftmax_m0.15_linear_bn_1e-2.json \
#    $data/voxceleb_train_combined_no_sil/train $data/voxceleb_train_combined_no_sil/train/spklist \
#    $data/voxceleb_train_combined_no_sil/softmax_valid $data/voxceleb_train_combined_no_sil/train/spklist \
#    $nnetdir
#
#nnetdir=$exp/xvector_nnet_tdnn_arcsoftmax_m0.20_linear_bn_1e-2
#nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_arcsoftmax_m0.20_linear_bn_1e-2.json \
#    $data/voxceleb_train_combined_no_sil/train $data/voxceleb_train_combined_no_sil/train/spklist \
#    $data/voxceleb_train_combined_no_sil/softmax_valid $data/voxceleb_train_combined_no_sil/train/spklist \
#    $nnetdir
#
#nnetdir=$exp/xvector_nnet_tdnn_arcsoftmax_m0.25_linear_bn_1e-2
#nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_arcsoftmax_m0.25_linear_bn_1e-2.json \
#    $data/voxceleb_train_combined_no_sil/train $data/voxceleb_train_combined_no_sil/train/spklist \
#    $data/voxceleb_train_combined_no_sil/softmax_valid $data/voxceleb_train_combined_no_sil/train/spklist \
#    $nnetdir
#
#nnetdir=$exp/xvector_nnet_tdnn_arcsoftmax_m0.30_linear_bn_1e-2
#nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_arcsoftmax_m0.30_linear_bn_1e-2.json \
#    $data/voxceleb_train_combined_no_sil/train $data/voxceleb_train_combined_no_sil/train/spklist \
#    $data/voxceleb_train_combined_no_sil/softmax_valid $data/voxceleb_train_combined_no_sil/train/spklist \
#    $nnetdir
#
#nnetdir=$exp/xvector_nnet_tdnn_arcsoftmax_m0.35_linear_bn_1e-2
#nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_arcsoftmax_m0.35_linear_bn_1e-2.json \
#    $data/voxceleb_train_combined_no_sil/train $data/voxceleb_train_combined_no_sil/train/spklist \
#    $data/voxceleb_train_combined_no_sil/softmax_valid $data/voxceleb_train_combined_no_sil/train/spklist \
#    $nnetdir
#
#nnetdir=$exp/xvector_nnet_tdnn_arcsoftmax_m0.40_linear_bn_1e-2
#nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_arcsoftmax_m0.40_linear_bn_1e-2.json \
#    $data/voxceleb_train_combined_no_sil/train $data/voxceleb_train_combined_no_sil/train/spklist \
#    $data/voxceleb_train_combined_no_sil/softmax_valid $data/voxceleb_train_combined_no_sil/train/spklist \
#    $nnetdir


## Add "Ring Loss"
#nnetdir=$exp/xvector_nnet_tdnn_amsoftmax_m0.20_linear_bn_1e-2_r0.01
#nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_amsoftmax_m0.20_linear_bn_1e-2_r0.01.json \
#    $data/voxceleb_train_combined_no_sil/train $data/voxceleb_train_combined_no_sil/train/spklist \
#    $data/voxceleb_train_combined_no_sil/softmax_valid $data/voxceleb_train_combined_no_sil/train/spklist \
#    $nnetdir

#nnetdir=$exp/xvector_nnet_tdnn_amsoftmax_m0.20_linear_bn_fn30_1e-2
#nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_amsoftmax_m0.20_linear_bn_fn30_1e-2.json \
#    $data/voxceleb_train_combined_no_sil/train $data/voxceleb_train_combined_no_sil/train/spklist \
#    $data/voxceleb_train_combined_no_sil/softmax_valid $data/voxceleb_train_combined_no_sil/train/spklist \
#    $nnetdir
#
## Add "MHE"
#nnetdir=$exp/xvector_nnet_tdnn_amsoftmax_m0.20_linear_bn_1e-2_mhe0.01
#nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training true nnet_conf/tdnn_amsoftmax_m0.20_linear_bn_1e-2_mhe0.01.json \
#    $data/voxceleb_train_combined_no_sil/train $data/voxceleb_train_combined_no_sil/train/spklist \
#    $data/voxceleb_train_combined_no_sil/softmax_valid $data/voxceleb_train_combined_no_sil/train/spklist \
#    $nnetdir


# Add attention
# nnetdir=$exp/xvector_nnet_tdnn_amsoftmax_m0.20_linear_bn_1e-2_tdnn4_att
# nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_amsoftmax_m0.20_linear_bn_1e-2_tdnn4_att.json \
#     $data/voxceleb_train_combined_no_sil/train $data/voxceleb_train_combined_no_sil/train/spklist \
#     $data/voxceleb_train_combined_no_sil/softmax_valid $data/voxceleb_train_combined_no_sil/train/spklist \
#     $nnetdir


exit 1
echo
fi

 nnetdir=exp_eftdnn2_a/xvector_nnet_tdnn_asoftmax_m4_linear_bn_1e-2
 checkpoint='last'

if [ $stage -le 8 ]; then
 # if you want to change GPU configuration, use this:
 #  gpu_bool=true
 #  gpu_init=0
 #  gpu_nj=8
 #  local/tf/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --use_gpu ${gpu_bool} --         gpu_index ${gpu_init} --nj ${gpu_nj} \
 #    $nnet_dir data/sre18_major \
 #    ${nnet_dir}/xvectors_sre18_major

  # Extract the embeddings for SRE (213111, SRE + SRE18), we separate it into sre_combined_without_sre16 (124349 utterances) and sre16_sre18_combined (88762 utterances)
  # SRE16 + SRE18 combined
  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj 80 --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "tdnn6_dense" \
    $nnetdir $data/sre16_sre18_combined $nnetdir/xvectors_sre16_sre18_combined

  # SRE without sre16
  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj 70 --use-gpu false --checkpoint        $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "tdnn6_dense" \
    $nnetdir $data/sre_combined_without_sre16 $nnetdir/xvectors_sre_combined_without_sre16

  utils/combine_data.sh data/sre_sre18_combined data/sre_combined data/sre16_sre18_combined
  mkdir $nnetdir/xvectors_sre_sre18_combined
  cat $nnetdir/xvectors_sre16_sre18_combined/xvector.scp $nnetdir/xvectors_sre_combined_without_sre16/xvector.scp > $nnetdir/xvectors_sre_sre18_combined/xvector.scp

  # The SRE18 major is an unlabeled dataset.  This is useful for things like centering, whitening and
  # score normalization.
  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj 80 --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "tdnn6_dense" \
    $nnetdir $data/sre18_major $nnetdir/xvectors_sre18_major

  # sre18_eval_test
  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj 80 --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "tdnn6_dense" \
    $nnetdir $data/sre18_eval_test $nnetdir/xvectors_sre18_eval_test

  # sre18_eval_enroll
  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj 80 --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "tdnn6_dense" \
    $nnetdir $data/sre18_eval_enroll $nnetdir/xvectors_sre18_eval_enroll

  # sre18_dev_test
  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj 80 --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "tdnn6_dense" \
    $nnetdir $data/sre18_dev_test $nnetdir/xvectors_sre18_dev_test

  # sre18_dev_enroll
  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj 80 --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "tdnn6_dense" \
    $nnetdir $data/sre18_dev_enroll $nnetdir/xvectors_sre18_dev_enroll

  # sre19_eval_test
  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj 80 --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "tdnn6_dense" \
    $nnetdir $data/sre19_eval_test $nnetdir/xvectors_sre19_eval_test

  # sre19_eval_enroll
  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj 80 --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "tdnn6_dense" \
    $nnetdir $data/sre19_eval_enroll $nnetdir/xvectors_sre19_eval_enroll
  exit
fi

data_for_lda=sre_sre18_combined
data_for_plda=$data_for_lda #same as lda
data_for_adapt=sre_sre18_combined

if [ $stage -le 9 ]; then
  # Compute the mean vector for centering the evaluation xvectors.
  $train_cmd ${nnet_dir}/xvectors_sre18_major/log/compute_mean.log \
    ivector-mean scp:${nnet_dir}/xvectors_sre18_major/xvector.scp \
    ${nnet_dir}/xvectors_sre18_major/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=150
  $train_cmd ${nnet_dir}/xvectors_$data_for_lda/log/lda.log \
    ivector-compute-lplda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:${nnet_dir}/xvectors_$data_for_lda/xvector.scp ark:- |" \
    ark:data/$data_for_lda/utt2spk ${nnet_dir}/xvectors_$data_for_lda/transform.mat || exit 1;

  # Train an out-of-domain PLDA model.
  $train_cmd ${nnet_dir}/xvectors_$data_for_plda/log/plda.log \
    ivector-compute-plda ark:data/$data_for_plda/spk2utt \
    "ark:ivector-subtract-global-mean scp:${nnet_dir}/xvectors_$data_for_plda/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_$data_for_plda/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    ${nnet_dir}/xvectors_$data_for_plda/plda || exit 1;

  # Here we adapt the out-of-domain PLDA model to SRE18 major, a pile
  # of unlabeled in-domain data.  In the future, we will include a clustering
  # based approach for domain adaptation, which tends to work better.
  $train_cmd ${nnet_dir}/xvectors_$data_for_adapt/log/plda_adapt.log \
    ivector-adapt-plda --within-covar-scale=0.75 --between-covar-scale=0.25 \
    ${nnet_dir}/xvectors_$data_for_adapt/plda \
    "ark:ivector-subtract-global-mean scp:${nnet_dir}/xvectors_sre18_major/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_$data_for_adapt/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    ${nnet_dir}/xvectors_$data_for_adapt/plda_adapt || exit 1;
fi




nnetdir=exp_kftdnn_am/xvector_nnet_tdnn_amsoftmax_m0.20_linear_bn_1e-2
  ##SRE18 DEV SCORING
echo "sre18 dev scoring\n"
if [ $stage -le 10 ]; then
  # Get sre18 dev results using the out-of-domain PLDA model.
  $train_cmd ${nnet_dir}/scores/log/sre18_dev_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:${nnet_dir}/xvectors_sre18_dev_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${nnet_dir}/xvectors_sre16_sre18_combined/plda - |" \
    "ark:ivector-mean ark:data/sre18_dev_enroll/spk2utt scp:${nnet_dir}/xvectors_sre18_dev_enroll/xvector.scp ark:- | ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec ark:- ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec scp:${nnet_dir}/xvectors_sre18_dev_test/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sre18_dev_trials' | cut -d\  --fields=1,2 |" ${nnet_dir}/scores/sre18_dev_scores || exit 1;

  ./tools/kaldiscore2nistscore.sh ${nnet_dir}/scores/sre18_dev_scores ${nnet_dir}/scores/sre18_dev_scores_nist
  python tools/cts_challenge_scoring_software/sre18_submission_scorer.py -o ${nnet_dir}/scores/sre18_dev_scores_nist -l /liry_tf/Data/SRE19/LDC2019E59/dev/docs/sre18_dev_trials.tsv -r /liry_tf/Data/SRE19/LDC2019E59/dev/docs/sre18_dev_trial_key.tsv
  pooled_eer=$(paste $sre18_dev_trials ${nnet_dir}/scores/sre18_dev_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "SRE18 DEV Using Out-of-Domain PLDA, EER: Pooled ${pooled_eer}%"
fi

if [ $stage -le 11 ]; then
  # Get sre18 dev results using the adapted PLDA model.
  $train_cmd ${nnet_dir}/scores/log/sre18_dev_scoring_adapt.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:${nnet_dir}/xvectors_sre18_dev_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${nnet_dir}/xvectors_sre18_major/plda_adapt - |" \
    "ark:ivector-mean ark:data/sre18_dev_enroll/spk2utt scp:${nnet_dir}/xvectors_sre18_dev_enroll/xvector.scp ark:- | ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec ark:- ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec scp:${nnet_dir}/xvectors_sre18_dev_test/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sre18_dev_trials' | cut -d\  --fields=1,2 |" ${nnet_dir}/scores/sre18_dev_scores_adapt || exit 1;

  ./tools/kaldiscore2nistscore.sh ${nnet_dir}/scores/sre18_dev_scores_adapt ${nnet_dir}/scores/sre18_dev_scores_adapt_nist
  python tools/cts_challenge_scoring_software/sre18_submission_scorer.py -o ${nnet_dir}/scores/sre18_dev_scores_adapt_nist -l /liry_tf/Data/SRE19/LDC2019E59/dev/docs/sre18_dev_trials.tsv -r /liry_tf/Data/SRE19/LDC2019E59/dev/docs/sre18_dev_trial_key.tsv
  pooled_eer=$(paste $sre18_dev_trials ${nnet_dir}/scores/sre18_dev_scores_adapt | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "SRE18 DEV Using Adapted PLDA, EER: Pooled ${pooled_eer}%"
fi

sre18_dev_enroll_list=data/sre18_dev_enroll/spk
sre18_dev_test_list=data/sre18_dev_test/spk
cut -d" " data/sre18_dev_enroll/spk2utt -f1 > ${sre18_dev_enroll_list}
cut -d" " data/sre18_dev_test/spk2utt -f1 > ${sre18_dev_test_list}

if [ $stage -le 12 ]; then
  # Get sre18 dev results using the out-of-domain PLDA model.
  $train_cmd ${nnet_dir}/scores/log/sre18_dev_scoring_matrix.log \
    ivector-plda-scoring-matrix --normalize-length=true \
    --num-utts=ark:${nnet_dir}/xvectors_sre18_dev_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${nnet_dir}/xvectors_sre16_sre18_combined/plda - |" \
    "ark:ivector-mean ark:data/sre18_dev_enroll/spk2utt scp:${nnet_dir}/xvectors_sre18_dev_enroll/xvector.scp ark:- | ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec ark:- ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec scp:${nnet_dir}/xvectors_sre18_dev_test/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $sre18_dev_enroll_list $sre18_dev_test_list ${nnet_dir}/scores/sre18_dev_scores_matrix || exit 1;
fi

if [ $stage -le 13 ]; then
  # Get sre18 dev results using the adapted PLDA model.
  $train_cmd ${nnet_dir}/scores/log/sre18_dev_scoring_adapt_matrix.log \
    ivector-plda-scoring-matrix --normalize-length=true \
    --num-utts=ark:${nnet_dir}/xvectors_sre18_dev_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${nnet_dir}/xvectors_sre18_major/plda_adapt - |" \
    "ark:ivector-mean ark:data/sre18_dev_enroll/spk2utt scp:${nnet_dir}/xvectors_sre18_dev_enroll/xvector.scp ark:- | ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec ark:- ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec scp:${nnet_dir}/xvectors_sre18_dev_test/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $sre18_dev_enroll_list $sre18_dev_test_list ${nnet_dir}/scores/sre18_dev_scores_adapt_matrix || exit 1;
fi

if [ $stage -le 14 ]; then
  # Get sre18 dev results using the out-of-domain PLDA model.
  $train_cmd ${nnet_dir}/scores/log/sre18_dev_enroll_scoring_matrix_snorm.log \
    ivector-plda-scoring-matrix --normalize-length=true \
    --num-utts=ark:${nnet_dir}/xvectors_sre18_dev_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${nnet_dir}/xvectors_sre16_sre18_combined/plda - |" \
    "ark:ivector-mean ark:data/sre18_dev_enroll/spk2utt scp:${nnet_dir}/xvectors_sre18_dev_enroll/xvector.scp ark:- | ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec ark:- ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec scp:${nnet_dir}/xvectors_sre18_major/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    ${sre18_dev_enroll_list} ${sre18_major_list} ${nnet_dir}/scores/sre18_dev_enroll_scores_matrix_majorsnorm || exit 1;
fi

if [ $stage -le 15 ]; then
  # Get sre18 dev results using the adapted PLDA model.
  $train_cmd ${nnet_dir}/scores/log/sre18_dev_enroll_scoring_adapt_matrix_snorm.log \
    ivector-plda-scoring-matrix --normalize-length=true \
    --num-utts=ark:${nnet_dir}/xvectors_sre18_dev_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${nnet_dir}/xvectors_sre18_major/plda_adapt - |" \
    "ark:ivector-mean ark:data/sre18_dev_enroll/spk2utt scp:${nnet_dir}/xvectors_sre18_dev_enroll/xvector.scp ark:- | ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec ark:- ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec scp:${nnet_dir}/xvectors_sre18_major/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    ${sre18_dev_enroll_list} ${sre18_major_list} ${nnet_dir}/scores/sre18_dev_enroll_scores_adapt_matrix_majorsnorm || exit 1;
fi

if [ $stage -le 16 ]; then
  # Get sre18 dev results using the out-of-domain PLDA model.
  $train_cmd ${nnet_dir}/scores/log/sre18_dev_test_scoring_matrix_snorm.log \
    ivector-plda-scoring-matrix --normalize-length=true \
    --num-utts=ark:${nnet_dir}/xvectors_sre18_dev_test/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${nnet_dir}/xvectors_sre16_sre18_combined/plda - |" \
    "ark:ivector-mean ark:data/sre18_dev_test/spk2utt scp:${nnet_dir}/xvectors_sre18_dev_test/xvector.scp ark:- | ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec ark:- ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec scp:${nnet_dir}/xvectors_sre18_major/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    ${sre18_dev_test_list} ${sre18_major_list} ${nnet_dir}/scores/sre18_dev_test_scores_matrix_majorsnorm || exit 1;
fi

if [ $stage -le 17 ]; then
  # Get sre18 dev results using the adapted PLDA model.
  $train_cmd ${nnet_dir}/scores/log/sre18_dev_test_scoring_adapt_matrix_snorm.log \
    ivector-plda-scoring-matrix --normalize-length=true \
    --num-utts=ark:${nnet_dir}/xvectors_sre18_dev_test/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${nnet_dir}/xvectors_sre18_major/plda_adapt - |" \
    "ark:ivector-mean ark:data/sre18_dev_test/spk2utt scp:${nnet_dir}/xvectors_sre18_dev_test/xvector.scp ark:- | ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec ark:- ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec scp:${nnet_dir}/xvectors_sre18_major/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    ${sre18_dev_test_list} ${sre18_major_list} ${nnet_dir}/scores/sre18_dev_test_scores_adapt_matrix_majorsnorm || exit 1;
fi

if [ $stage -le 18 ]; then
  python tools/s-norm-get-enroll.py --score-file ${nnet_dir}/scores/sre18_dev_enroll_scores_matrix_majorsnorm --enroll-file ${nnet_dir}/scores/sre18_dev_enroll.snorm

  python tools/s-norm-get-enroll.py --score-file ${nnet_dir}/scores/sre18_dev_test_scores_matrix_majorsnorm --enroll-file ${nnet_dir}/scores/sre18_dev_test.snorm

  python tools/s-norm-from-enroll.py --score-file ${nnet_dir}/scores/sre18_dev_scores_matrix --snorm-file ${nnet_dir}/scores/sre18_dev_scores_matrix_snorm --enroll-file ${nnet_dir}/scores/sre18_dev_enroll.snorm --eval-file ${nnet_dir}/scores/sre18_dev_test.snorm

  python tools/matrix-kaldi.py --matrix-file ${nnet_dir}/scores/sre18_dev_scores_matrix_snorm --trial-file /mnt/workspace/project/SRE19/LDC2019E59/dev/docs/sre18_dev_trials.tsv --kaldi-file ${nnet_dir}/scores/sre18_dev_scores_snorm.tsv

  python tools/cts_challenge_scoring_software/sre18_submission_scorer.py -o ${nnet_dir}/scores/sre18_dev_scores_snorm.tsv -l /liry_tf/Data/SRE19/LDC2019E59/dev/docs/sre18_dev_trials.tsv -r /liry_tf/Data/SRE19/LDC2019E59/dev/docs/sre18_dev_trial_key.tsv
fi

if [ $stage -le 19 ]; then
  python tools/s-norm-get-enroll.py --score-file ${nnet_dir}/scores/sre18_dev_enroll_scores_adapt_matrix_majorsnorm --enroll-file ${nnet_dir}/scores/sre18_dev_enroll_adapt.snorm

  python tools/s-norm-get-enroll.py --score-file ${nnet_dir}/scores/sre18_dev_test_scores_adapt_matrix_majorsnorm --enroll-file ${nnet_dir}/scores/sre18_dev_test_adapt.snorm

  python tools/s-norm-from-enroll.py --score-file ${nnet_dir}/scores/sre18_dev_scores_adapt_matrix --snorm-file ${nnet_dir}/scores/sre18_dev_scores_adapt_matrix_snorm --enroll-file ${nnet_dir}/scores/sre18_dev_enroll_adapt.snorm --eval-file ${nnet_dir}/scores/sre18_dev_test_adapt.snorm

  python tools/matrix-kaldi.py --matrix-file ${nnet_dir}/scores/sre18_dev_scores_adapt_matrix_snorm --trial-file /mnt/workspace/project/SRE19/LDC2019E59/dev/docs/sre18_dev_trials.tsv --kaldi-file ${nnet_dir}/scores/sre18_dev_scores_adapt_snorm.tsv

  python tools/cts_challenge_scoring_software/sre18_submission_scorer.py -o ${nnet_dir}/scores/sre18_dev_scores_adapt_snorm.tsv -l /liry_tf/Data/SRE19/LDC2019E59/dev/docs/sre18_dev_trials.tsv -r /liry_tf/Data/SRE19/LDC2019E59/dev/docs/sre18_dev_trial_key.tsv
fi



  ##SRE18 EVAL SCORING
echo "sre18 eval scoring\n"
if [ $stage -le 20 ]; then
  # Get sre18 eval results using the out-of-domain PLDA model.
  $train_cmd ${nnet_dir}/scores/log/sre18_eval_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:${nnet_dir}/xvectors_sre18_eval_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${nnet_dir}/xvectors_sre16_sre18_combined/plda - |" \
    "ark:ivector-mean ark:data/sre18_eval_enroll/spk2utt scp:${nnet_dir}/xvectors_sre18_eval_enroll/xvector.scp ark:- | ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec ark:- ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec scp:${nnet_dir}/xvectors_sre18_eval_test/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sre18_eval_trials' | cut -d\  --fields=1,2 |" ${nnet_dir}/scores/sre18_eval_scores || exit 1;

  ./tools/kaldiscore2nistscore.sh ${nnet_dir}/scores/sre18_eval_scores ${nnet_dir}/scores/sre18_eval_scores_nist
  python tools/cts_challenge_scoring_software/sre18_submission_scorer.py -o ${nnet_dir}/scores/sre18_eval_scores_nist -l /liry_tf/Data/SRE19/LDC2019E59/eval/docs/sre18_eval_trials.tsv -r /liry_tf/Data/SRE19/LDC2019E59/eval/docs/sre18_eval_trial_key.tsv
  pooled_eer=$(paste $sre18_eval_trials ${nnet_dir}/scores/sre18_eval_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "SRE18 EVAL Using Out-of-Domain PLDA, EER: Pooled ${pooled_eer}%"
fi

if [ $stage -le 21 ]; then
  # Get sre18 eval results using the adapted PLDA model.
  $train_cmd ${nnet_dir}/scores/log/sre18_eval_scoring_adapt.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:${nnet_dir}/xvectors_sre18_eval_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${nnet_dir}/xvectors_sre18_major/plda_adapt - |" \
    "ark:ivector-mean ark:data/sre18_eval_enroll/spk2utt scp:${nnet_dir}/xvectors_sre18_eval_enroll/xvector.scp ark:- | ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec ark:- ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec scp:${nnet_dir}/xvectors_sre18_eval_test/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sre18_eval_trials' | cut -d\  --fields=1,2 |" ${nnet_dir}/scores/sre18_eval_scores_adapt || exit 1;

  ./tools/kaldiscore2nistscore.sh ${nnet_dir}/scores/sre18_eval_scores_adapt ${nnet_dir}/scores/sre18_eval_scores_adapt_nist
  python tools/cts_challenge_scoring_software/sre18_submission_scorer.py -o ${nnet_dir}/scores/sre18_eval_scores_adapt_nist -l /liry_tf/Data/SRE19/LDC2019E59/eval/docs/sre18_eval_trials.tsv -r /liry_tf/Data/SRE19/LDC2019E59/eval/docs/sre18_eval_trial_key.tsv
  pooled_eer=$(paste $sre18_eval_trials ${nnet_dir}/scores/sre18_eval_scores_adapt | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "SRE18 EVAL Using Adapted PLDA, EER: Pooled ${pooled_eer}%"
fi

sre18_eval_enroll_list=data/sre18_eval_enroll/spk
sre18_eval_test_list=data/sre18_eval_test/spk
cut -d" " data/sre18_eval_enroll/spk2utt -f1 > ${sre18_eval_enroll_list}
cut -d" " data/sre18_eval_test/spk2utt -f1 > ${sre18_eval_test_list}

if [ $stage -le 22 ]; then
  # Get sre18 eval results using the out-of-domain PLDA model.
  $train_cmd ${nnet_dir}/scores/log/sre18_eval_scoring_matrix.log \
    ivector-plda-scoring-matrix --normalize-length=true \
    --num-utts=ark:${nnet_dir}/xvectors_sre18_eval_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${nnet_dir}/xvectors_sre16_sre18_combined/plda - |" \
    "ark:ivector-mean ark:data/sre18_eval_enroll/spk2utt scp:${nnet_dir}/xvectors_sre18_eval_enroll/xvector.scp ark:- | ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec ark:- ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec scp:${nnet_dir}/xvectors_sre18_eval_test/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $sre18_eval_enroll_list $sre18_eval_test_list ${nnet_dir}/scores/sre18_eval_scores_matrix || exit 1;
fi

if [ $stage -le 23 ]; then
  # Get sre18 eval results using the adapted PLDA model.
  $train_cmd ${nnet_dir}/scores/log/sre18_eval_scoring_adapt_matrix.log \
    ivector-plda-scoring-matrix --normalize-length=true \
    --num-utts=ark:${nnet_dir}/xvectors_sre18_eval_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${nnet_dir}/xvectors_sre18_major/plda_adapt - |" \
    "ark:ivector-mean ark:data/sre18_eval_enroll/spk2utt scp:${nnet_dir}/xvectors_sre18_eval_enroll/xvector.scp ark:- | ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec ark:- ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec scp:${nnet_dir}/xvectors_sre18_eval_test/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $sre18_eval_enroll_list $sre18_eval_test_list ${nnet_dir}/scores/sre18_eval_scores_adapt_matrix || exit 1;
fi

if [ $stage -le 24 ]; then
  # Get sre18 eval results using the out-of-domain PLDA model.
  $train_cmd ${nnet_dir}/scores/log/sre18_eval_enroll_scoring_matrix_snorm.log \
    ivector-plda-scoring-matrix --normalize-length=true \
    --num-utts=ark:${nnet_dir}/xvectors_sre18_eval_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${nnet_dir}/xvectors_sre16_sre18_combined/plda - |" \
    "ark:ivector-mean ark:data/sre18_eval_enroll/spk2utt scp:${nnet_dir}/xvectors_sre18_eval_enroll/xvector.scp ark:- | ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec ark:- ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec scp:${nnet_dir}/xvectors_sre18_major/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    ${sre18_eval_enroll_list} ${sre18_major_list} ${nnet_dir}/scores/sre18_eval_enroll_scores_matrix_majorsnorm || exit 1;
fi

if [ $stage -le 25 ]; then
  # Get sre18 eval results using the adapted PLDA model.
  $train_cmd ${nnet_dir}/scores/log/sre18_eval_enroll_scoring_adapt_matrix_snorm.log \
    ivector-plda-scoring-matrix --normalize-length=true \
    --num-utts=ark:${nnet_dir}/xvectors_sre18_eval_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${nnet_dir}/xvectors_sre18_major/plda_adapt - |" \
    "ark:ivector-mean ark:data/sre18_eval_enroll/spk2utt scp:${nnet_dir}/xvectors_sre18_eval_enroll/xvector.scp ark:- | ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec ark:- ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec scp:${nnet_dir}/xvectors_sre18_major/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    ${sre18_eval_enroll_list} ${sre18_major_list} ${nnet_dir}/scores/sre18_eval_enroll_scores_adapt_matrix_majorsnorm || exit 1;
fi

if [ $stage -le 26 ]; then
  # Get sre19 eval results using the out-of-domain PLDA model.
  $train_cmd ${nnet_dir}/scores/log/sre18_eval_test_scoring_matrix_snorm.log \
    ivector-plda-scoring-matrix --normalize-length=true \
    --num-utts=ark:${nnet_dir}/xvectors_sre18_eval_test/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${nnet_dir}/xvectors_sre16_sre18_combined/plda - |" \
    "ark:ivector-mean ark:data/sre18_eval_test/spk2utt scp:${nnet_dir}/xvectors_sre18_eval_test/xvector.scp ark:- | ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec ark:- ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec scp:${nnet_dir}/xvectors_sre18_major/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    ${sre18_eval_test_list} ${sre18_major_list} ${nnet_dir}/scores/sre18_eval_test_scores_matrix_majorsnorm || exit 1;
fi

if [ $stage -le 27 ]; then
  # Get sre18 eval results using the adapted PLDA model.
  $train_cmd ${nnet_dir}/scores/log/sre18_eval_test_scoring_adapt_matrix_snorm.log \
    ivector-plda-scoring-matrix --normalize-length=true \
    --num-utts=ark:${nnet_dir}/xvectors_sre18_eval_test/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${nnet_dir}/xvectors_sre18_major/plda_adapt - |" \
    "ark:ivector-mean ark:data/sre18_eval_test/spk2utt scp:${nnet_dir}/xvectors_sre18_eval_test/xvector.scp ark:- | ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec ark:- ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec scp:${nnet_dir}/xvectors_sre18_major/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    ${sre18_eval_test_list} ${sre18_major_list} ${nnet_dir}/scores/sre18_eval_test_scores_adapt_matrix_majorsnorm || exit 1;
fi

if [ $stage -le 28 ]; then
  python tools/s-norm-get-enroll.py --score-file ${nnet_dir}/scores/sre18_eval_enroll_scores_matrix_majorsnorm --enroll-file ${nnet_dir}/scores/sre18_eval_enroll.snorm

  python tools/s-norm-get-enroll.py --score-file ${nnet_dir}/scores/sre18_eval_test_scores_matrix_majorsnorm --enroll-file ${nnet_dir}/scores/sre18_eval_test.snorm

  python tools/s-norm-from-enroll.py --score-file ${nnet_dir}/scores/sre18_eval_scores_matrix --snorm-file ${nnet_dir}/scores/sre18_eval_scores_matrix_snorm --enroll-file ${nnet_dir}/scores/sre18_eval_enroll.snorm --eval-file ${nnet_dir}/scores/sre18_eval_test.snorm

  python tools/matrix-kaldi.py --matrix-file ${nnet_dir}/scores/sre18_eval_scores_matrix_snorm --trial-file /mnt/workspace/project/SRE19/LDC2019E59/eval/docs/sre18_eval_trials.tsv --kaldi-file ${nnet_dir}/scores/sre18_eval_scores_snorm.tsv

  python tools/cts_challenge_scoring_software/sre18_submission_scorer.py -o ${nnet_dir}/scores/sre18_eval_scores_snorm.tsv -l /liry_tf/Data/SRE19/LDC2019E59/eval/docs/sre18_eval_trials.tsv -r /liry_tf/Data/SRE19/LDC2019E59/eval/docs/sre18_eval_trial_key.tsv
fi

if [ $stage -le 29 ]; then
  python tools/s-norm-get-enroll.py --score-file ${nnet_dir}/scores/sre18_eval_enroll_scores_adapt_matrix_majorsnorm --enroll-file ${nnet_dir}/scores/sre18_eval_enroll_adapt.snorm

  python tools/s-norm-get-enroll.py --score-file ${nnet_dir}/scores/sre18_eval_test_scores_adapt_matrix_majorsnorm --enroll-file ${nnet_dir}/scores/sre18_eval_test_adapt.snorm

  python tools/s-norm-from-enroll.py --score-file ${nnet_dir}/scores/sre18_eval_scores_adapt_matrix --snorm-file ${nnet_dir}/scores/sre18_eval_scores_adapt_matrix_snorm --enroll-file ${nnet_dir}/scores/sre18_eval_enroll_adapt.snorm --eval-file ${nnet_dir}/scores/sre18_eval_test_adapt.snorm

  python tools/matrix-kaldi.py --matrix-file ${nnet_dir}/scores/sre18_eval_scores_adapt_matrix_snorm --trial-file /mnt/workspace/project/SRE19/LDC2019E59/eval/docs/sre18_eval_trials.tsv --kaldi-file ${nnet_dir}/scores/sre18_eval_scores_adapt_snorm.tsv

  python tools/cts_challenge_scoring_software/sre18_submission_scorer.py -o ${nnet_dir}/scores/sre18_eval_scores_adapt_snorm.tsv -l /liry_tf/Data/SRE19/LDC2019E59/eval/docs/sre18_eval_trials.tsv -r /liry_tf/Data/SRE19/LDC2019E59/eval/docs/sre18_eval_trial_key.tsv
fi









  ##SRE19 EVAL SCORING
echo "sre19 eval scoring\n"
if [ $stage -le 30 ]; then
  # Get sre19 eval results using the out-of-domain PLDA model.
  $train_cmd ${nnet_dir}/scores/log/sre19_eval_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:${nnet_dir}/xvectors_sre19_eval_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${nnet_dir}/xvectors_sre16_sre18_combined/plda - |" \
    "ark:ivector-mean ark:data/sre19_eval_enroll/spk2utt scp:${nnet_dir}/xvectors_sre19_eval_enroll/xvector.scp ark:- | ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec ark:- ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec scp:${nnet_dir}/xvectors_sre19_eval_test/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sre19_eval_trials' | cut -d\  --fields=1,2 |" ${nnet_dir}/scores/sre19_eval_scores || exit 1;
fi

if [ $stage -le 31 ]; then
  # Get sre18 eval results using the adapted PLDA model.
  $train_cmd ${nnet_dir}/scores/log/sre19_eval_scoring_adapt.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:${nnet_dir}/xvectors_sre19_eval_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${nnet_dir}/xvectors_sre18_major/plda_adapt - |" \
    "ark:ivector-mean ark:data/sre19_eval_enroll/spk2utt scp:${nnet_dir}/xvectors_sre19_eval_enroll/xvector.scp ark:- | ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec ark:- ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec scp:${nnet_dir}/xvectors_sre19_eval_test/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sre19_eval_trials' | cut -d\  --fields=1,2 |" ${nnet_dir}/scores/sre19_eval_scores_adapt || exit 1;
fi

sre19_eval_enroll_list=data/sre19_eval_enroll/spk
sre19_eval_test_list=data/sre19_eval_test/spk
cut -d" " data/sre19_eval_enroll/spk2utt -f1 > ${sre19_eval_enroll_list}
cut -d" " data/sre19_eval_test/spk2utt -f1 > ${sre19_eval_test_list}

if [ $stage -le 32 ]; then
  # Get sre19 eval results using the out-of-domain PLDA model.
  $train_cmd ${nnet_dir}/scores/log/sre19_eval_scoring_matrix.log \
    ivector-plda-scoring-matrix --normalize-length=true \
    --num-utts=ark:${nnet_dir}/xvectors_sre19_eval_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${nnet_dir}/xvectors_sre16_sre18_combined/plda - |" \
    "ark:ivector-mean ark:data/sre19_eval_enroll/spk2utt scp:${nnet_dir}/xvectors_sre19_eval_enroll/xvector.scp ark:- | ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec ark:- ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec scp:${nnet_dir}/xvectors_sre19_eval_test/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    ${sre19_eval_enroll_list} ${sre19_eval_test_list} ${nnet_dir}/scores/sre19_eval_scores_matrix || exit 1;
fi

if [ $stage -le 33 ]; then
  # Get sre18 eval results using the adapted PLDA model.
  $train_cmd ${nnet_dir}/scores/log/sre19_eval_scoring_adapt_matrix.log \
    ivector-plda-scoring-matrix --normalize-length=true \
    --num-utts=ark:${nnet_dir}/xvectors_sre19_eval_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${nnet_dir}/xvectors_sre18_major/plda_adapt - |" \
    "ark:ivector-mean ark:data/sre19_eval_enroll/spk2utt scp:${nnet_dir}/xvectors_sre19_eval_enroll/xvector.scp ark:- | ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec ark:- ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec scp:${nnet_dir}/xvectors_sre19_eval_test/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    ${sre19_eval_enroll_list} ${sre19_eval_test_list} ${nnet_dir}/scores/sre19_eval_scores_adapt_matrix || exit 1;
fi

if [ $stage -le 34 ]; then
  # Get sre19 eval results using the out-of-domain PLDA model.
  $train_cmd ${nnet_dir}/scores/log/sre19_eval_enroll_scoring_matrix_snorm.log \
    ivector-plda-scoring-matrix --normalize-length=true \
    --num-utts=ark:${nnet_dir}/xvectors_sre19_eval_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${nnet_dir}/xvectors_sre16_sre18_combined/plda - |" \
    "ark:ivector-mean ark:data/sre19_eval_enroll/spk2utt scp:${nnet_dir}/xvectors_sre19_eval_enroll/xvector.scp ark:- | ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec ark:- ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec scp:${nnet_dir}/xvectors_sre18_major/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    ${sre19_eval_enroll_list} ${sre18_major_list} ${nnet_dir}/scores/sre19_eval_enroll_scores_matrix_majorsnorm || exit 1;
fi

if [ $stage -le 35 ]; then
  # Get sre18 eval results using the adapted PLDA model.
  $train_cmd ${nnet_dir}/scores/log/sre19_enroll_scoring_adapt_matrix_snorm.log \
    ivector-plda-scoring-matrix --normalize-length=true \
    --num-utts=ark:${nnet_dir}/xvectors_sre19_eval_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${nnet_dir}/xvectors_sre18_major/plda_adapt - |" \
    "ark:ivector-mean ark:data/sre19_eval_enroll/spk2utt scp:${nnet_dir}/xvectors_sre19_eval_enroll/xvector.scp ark:- | ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec ark:- ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec scp:${nnet_dir}/xvectors_sre18_major/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    ${sre19_eval_enroll_list} ${sre18_major_list} ${nnet_dir}/scores/sre19_eval_enroll_scores_adapt_matrix_majorsnorm || exit 1;
fi

if [ $stage -le 36 ]; then
  # Get sre19 eval results using the out-of-domain PLDA model.
  $train_cmd ${nnet_dir}/scores/log/sre19_eval_test_scoring_matrix_snorm.log \
    ivector-plda-scoring-matrix --normalize-length=true \
    --num-utts=ark:${nnet_dir}/xvectors_sre19_eval_test/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${nnet_dir}/xvectors_sre16_sre18_combined/plda - |" \
    "ark:ivector-mean ark:data/sre19_eval_test/spk2utt scp:${nnet_dir}/xvectors_sre19_eval_test/xvector.scp ark:- | ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec ark:- ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec scp:${nnet_dir}/xvectors_sre18_major/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    ${sre19_eval_test_list} ${sre18_major_list} ${nnet_dir}/scores/sre19_eval_test_scores_matrix_majorsnorm || exit 1;
fi

if [ $stage -le 37 ]; then
  # Get sre18 eval results using the adapted PLDA model.
  $train_cmd ${nnet_dir}/scores/log/sre19_eval_test_scoring_adapt_matrix_snorm.log \
    ivector-plda-scoring-matrix --normalize-length=true \
    --num-utts=ark:${nnet_dir}/xvectors_sre19_eval_test/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${nnet_dir}/xvectors_sre18_major/plda_adapt - |" \
    "ark:ivector-mean ark:data/sre19_eval_test/spk2utt scp:${nnet_dir}/xvectors_sre19_eval_test/xvector.scp ark:- | ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec ark:- ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${nnet_dir}/xvectors_sre18_major/mean.vec scp:${nnet_dir}/xvectors_sre18_major/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre16_sre18_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    ${sre19_eval_test_list} ${sre18_major_list} ${nnet_dir}/scores/sre19_eval_test_scores_adapt_matrix_majorsnorm || exit 1;
fi

if [ $stage -le 38 ]; then
  python tools/s-norm-get-enroll.py --score-file ${nnet_dir}/scores/sre19_eval_enroll_scores_matrix_majorsnorm --enroll-file ${nnet_dir}/scores/sre19_eval_enroll.snorm

  python tools/s-norm-get-enroll.py --score-file ${nnet_dir}/scores/sre19_eval_test_scores_matrix_majorsnorm --enroll-file ${nnet_dir}/scores/sre19_eval_test.snorm

  python tools/s-norm-from-enroll.py --score-file ${nnet_dir}/scores/sre19_eval_scores_matrix --snorm-file ${nnet_dir}/scores/sre19_eval_scores_matrix_snorm --enroll-file ${nnet_dir}/scores/sre19_eval_enroll.snorm --eval-file ${nnet_dir}/scores/sre19_eval_test.snorm

  python tools/matrix-kaldi.py --matrix-file ${nnet_dir}/scores/sre19_eval_scores_matrix_snorm --trial-file /mnt/workspace/project/SRE19/LDC2019E58/docs/sre19_cts_challenge_trials.tsv --kaldi-file ${nnet_dir}/scores/sre19_eval_scores_snorm.tsv
fi

if [ $stage -le 39 ]; then
  python tools/s-norm-get-enroll.py --score-file ${nnet_dir}/scores/sre19_eval_enroll_scores_adapt_matrix_majorsnorm --enroll-file ${nnet_dir}/scores/sre19_eval_enroll_adapt.snorm

  python tools/s-norm-get-enroll.py --score-file ${nnet_dir}/scores/sre19_eval_test_scores_adapt_matrix_majorsnorm --enroll-file ${nnet_dir}/scores/sre19_eval_test_adapt.snorm

  python tools/s-norm-from-enroll.py --score-file ${nnet_dir}/scores/sre19_eval_scores_adapt_matrix --snorm-file ${nnet_dir}/scores/sre19_eval_scores_adapt_matrix_snorm --enroll-file ${nnet_dir}/scores/sre19_eval_enroll_adapt.snorm --eval-file ${nnet_dir}/scores/sre19_eval_test_adapt.snorm

  python tools/matrix-kaldi.py --matrix-file ${nnet_dir}/scores/sre19_eval_scores_adapt_matrix_snorm --trial-file /mnt/workspace/project/SRE19/LDC2019E58/docs/sre19_cts_challenge_trials.tsv --kaldi-file ${nnet_dir}/scores/sre19_eval_scores_adapt_snorm.tsv
fi
