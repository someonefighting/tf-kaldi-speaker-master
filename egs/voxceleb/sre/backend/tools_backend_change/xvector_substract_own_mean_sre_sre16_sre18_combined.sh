#!/bin/bash

nnet_dir=$1

xvector_dir=$nnet_dir/xvectors_sre_combined_sre18/

[ ! -f $xvector_dir/xvector.scp.ori ] && cp $xvector_dir/xvector.scp $xvector_dir/xvector.scp.ori || exit 0
cp $xvector_dir/xvector.scp $xvector_dir/xvector.scp.`date "+%Y%m%d_%H%M%S"`.bak

grep sre16.*sre16.*sre16_sre18 $xvector_dir/xvector.scp > $xvector_dir/xvector_sre16.scp
grep sre18.*sre18.*sre16_sre18 $xvector_dir/xvector.scp > $xvector_dir/xvector_sre18.scp
grep -v sre16.*sre16.*sre16_sre18 $xvector_dir/xvector.scp > $xvector_dir/xvector_sre.scp.tmp
grep -v sre18.*sre18.*sre16_sre18 $xvector_dir/xvector_sre.scp.tmp > $xvector_dir/xvector_sre.scp

ivector-subtract-global-mean scp:$xvector_dir/xvector_sre16.scp ark,scp:$xvector_dir/xvector_sre16.sub.ark,$xvector_dir/xvector_sre16.sub.scp
ivector-subtract-global-mean scp:$xvector_dir/xvector_sre18.scp ark,scp:$xvector_dir/xvector_sre18.sub.ark,$xvector_dir/xvector_sre18.sub.scp
ivector-subtract-global-mean scp:$xvector_dir/xvector_sre.scp ark,scp:$xvector_dir/xvector_sre.sub.ark,$xvector_dir/xvector_sre.sub.scp

rm $xvector_dir/xvector.scp
cat $xvector_dir/xvector_sre.sub.scp >> $xvector_dir/xvector.scp
cat $xvector_dir/xvector_sre16.sub.scp >> $xvector_dir/xvector.scp
cat $xvector_dir/xvector_sre18.sub.scp >> $xvector_dir/xvector.scp
