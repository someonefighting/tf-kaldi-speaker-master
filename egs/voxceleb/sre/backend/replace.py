import os
import re
from os import path
import sys
utt2spk = sys.argv[1]+"/utt2spk"
feats = sys.argv[1]+"/feats.scp"
utt2num_frames = sys.argv[1]+"/utt2num_frames"
vad = sys.argv[1]+"/vad.scp"
wav = sys.argv[1]+"/wav.scp"
dict_sub = {}
dict_utt = {}
with open('sre16_model_spkid_mapping.ndx') as f:
    for line in f:
        line = line.strip()
        ut,spk = line.split()
        #spkid, uttid = ut.split('-')
        #print(spkid, uttid)
        dict_sub[ut] = spk

with open('sre18_model_spkid_mapping.ndx') as f:
    for line in f:
        line = line.strip()
        ut,spk = line.split()
        #spkid, uttid = ut.split('-')
        #print(spkid, uttid)
        dict_sub[ut] = spk

fp_out = open('utt2spk_new', 'w')
with open(utt2spk) as f:
    for line in f:
        line = line.strip()
        sp, su= line.split()
        if su in dict_sub:
            f1,f2,f3 = re.split('_|-', sp, 2)
            dict_utt[sp] = dict_sub[su]+'_'+f3
            fp_out.write(dict_utt[sp] + ' ' + dict_sub[su] + '\n')
        else:
            dict_utt[sp] = sp
            fp_out.write(line + '\n')

fp_out.close()
fp_out = open('feat_new', 'w')
with open(feats) as f:
    for line in f:
        line = line.strip()
        sp, su= line.split()
        fp_out.write(dict_utt[sp] + ' ' + su + '\n')
fp_out.close()
if path.isfile(vad):
    fp_out = open('vad_new', 'w')
    with open(vad) as f:
        for line in f:
            line = line.strip()
            sp, su= line.split()
            fp_out.write(dict_utt[sp] + ' ' + su + '\n')
    fp_out.close()
if path.isfile(utt2num_frames):
    fp_out = open('utt2num_frames_new', 'w')
    with open(utt2num_frames) as f:
        for line in f:
            line = line.strip()
            sp, su= line.split()
            fp_out.write(dict_utt[sp] + ' ' + su + '\n')
    fp_out.close()
if path.isfile(wav):
    fp_out = open('wav_new', 'w')
    with open(wav) as f:
        for line in f:
            line = line.strip()
            sp, su= line.split(' ',1)
            fp_out.write(dict_utt[sp] + ' ' + su + '\n')
    fp_out.close()
