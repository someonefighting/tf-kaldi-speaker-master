#!/bin/bash

data_dir="sre16_sre18_combined"
for d in $data_dir; do
  python replace.py $d
  sort -u utt2spk_new > $data_dir/utt2spk
  utils/utt2spk_to_spk2utt.pl $data_dir/utt2spk > $data_dir/spk2utt
  sort -u feat_new > $data_dir/feats.scp
  if [ -f "wav_new" ]; then
    sort -u wav_new > $data_dir/wav.scp
  fi
  if [ -f "vad_new" ]; then
    sort -u vad_new > $data_dir/vad.scp
  fi
  if [ -f "utt2num_frames_new" ]; then
    sort -u utt2num_frames_new > $data_dir/utt2num_frames
  fi
  utils/validate_data_dir.sh --no-text --no-wav $data_dir
done
