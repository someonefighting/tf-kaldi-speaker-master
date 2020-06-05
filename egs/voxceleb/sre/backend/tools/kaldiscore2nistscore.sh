#!/bin/bash


if [ $# != 2 ];then
  echo "USAGE: $0 kaldiscore nistscore"
  echo "e.g.: $0 sre19_eval_scores_adapt sre19_eval_system_output.tsv"
  exit 1;
fi

kaldiscore=$1
nistscore=$2

awk -F' ' '{print $1"\t"$2".sph\t""a\t"$3}' $1 \
  | sed  '1i\modelid\tsegmentid\tside\tLLR'  > $2

