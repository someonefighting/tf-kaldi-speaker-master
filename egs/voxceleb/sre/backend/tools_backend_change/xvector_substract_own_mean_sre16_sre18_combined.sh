#!/bin/bash

nnet_dir=$1

xvector_dir=$nnet_dir/xvectors_sre16_sre18_combined/

[ ! -f $xvector_dir/xvector.scp.ori ] && cp $xvector_dir/xvector.scp $xvector_dir/xvector.scp.ori || exit 0
cp $xvector_dir/xvector.scp $xvector_dir/xvector.scp.`date "+%Y%m%d_%H%M%S"`.bak

grep sre16.*sre16.*sre16_sre18 $xvector_dir/xvector.scp > $xvector_dir/xvector_sre16.scp
grep sre18.*sre18.*sre16_sre18 $xvector_dir/xvector.scp > $xvector_dir/xvector_sre18.scp

ivector-subtract-global-mean scp:$xvector_dir/xvector_sre16.scp ark,scp:$xvector_dir/xvector_sre16.sub.ark,$xvector_dir/xvector_sre16.sub.scp
ivector-subtract-global-mean scp:$xvector_dir/xvector_sre18.scp ark,scp:$xvector_dir/xvector_sre18.sub.ark,$xvector_dir/xvector_sre18.sub.scp

rm $xvector_dir/xvector.scp
cat $xvector_dir/xvector_sre16.sub.scp >> $xvector_dir/xvector.scp
cat $xvector_dir/xvector_sre18.sub.scp >> $xvector_dir/xvector.scp
