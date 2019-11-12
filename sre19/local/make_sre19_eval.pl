#!/usr/bin/perl
use warnings; #sed replacement for -w perl parameter
# Copyright 2017   David Snyder
# Apache 2.0
#

if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-SRE19-eval> <path-to-output>\n";
  print STDERR "e.g. $0 /export/corpora/SRE/LDC2019E58 data/\n";
  exit(1);
}

($db_base, $out_dir) = @ARGV;

# Handle enroll
$out_dir_enroll = "$out_dir/sre19_eval_enroll";
if (system("mkdir -p $out_dir_enroll")) {
  die "Error making directory $out_dir_enroll";
}

$tmp_dir_enroll = "$out_dir_enroll/tmp";
if (system("mkdir -p $tmp_dir_enroll") != 0) {
  die "Error making directory $tmp_dir_enroll";
}

open(SPKR, ">$out_dir_enroll/utt2spk") || die "Could not open the output file $out_dir_enroll/utt2spk";
open(WAV, ">$out_dir_enroll/wav.scp") || die "Could not open the output file $out_dir_enroll/wav.scp";
open(META, "<${db_base}/docs/sre19_cts_challenge_enrollment.tsv") or die "cannot open wav list";
%utt2fixedutt = ();
while (<META>) {
  $line = $_;
  @toks = split(" ", $line);
  $spk = $toks[0];
  $uttsph = $toks[1];
  $utt = (split("[./]", $uttsph))[0];
  if ($utt ne "segmentid") {
    print SPKR "${spk}-${utt} $spk\n";
    $utt2fixedutt{$utt} = "${spk}-${utt}";
  }
}

if (system("find $db_base/data/enrollment/ -name '*.sph' > $tmp_dir_enroll/sph.list") != 0) {
  die "Error getting list of sph files";
}

open(WAVLIST, "<$tmp_dir_enroll/sph.list") or die "cannot open wav list";

while(<WAVLIST>) {
  chomp;
  $sph = $_;
  @t = split("/",$sph);
  @t1 = split("[./]",$t[$#t]);
  $utt=$utt2fixedutt{$t1[0]};
  if ( $t1[1] eq "sph"){
  print WAV "$utt"," sph2pipe -f wav -p -c 1 $sph |\n";
  }else{
  print "no this wav format $t1[1].\n";
  }
}
close(WAV) || die;
close(SPKR) || die;

# Handle test
$out_dir_test= "$out_dir/sre19_eval_test";
if (system("mkdir -p $out_dir_test")) {
  die "Error making directory $out_dir_test";
}

$tmp_dir_test = "$out_dir_test/tmp";
if (system("mkdir -p $tmp_dir_test") != 0) {
  die "Error making directory $tmp_dir_test";
}

open(SPKR, ">$out_dir_test/utt2spk") || die "Could not open the output file $out_dir_test/utt2spk";
open(WAV, ">$out_dir_test/wav.scp") || die "Could not open the output file $out_dir_test/wav.scp";
open(TRIALS, ">$out_dir_test/trials") || die "Could not open the output file $out_dir_test/trials";

if (system("find $db_base/data/test/ -name '*.sph' > $tmp_dir_test/sph.list") != 0) {
  die "Error getting list of sph files";
}


open(KEY, "<${db_base}/docs/sre19_cts_challenge_trials.tsv") || die "Could not open trials file $db_base/docs/sre18_eval_trials.tsv.  It might be located somewhere else in your distribution.";
open(WAVLIST, "<$tmp_dir_test/sph.list") or die "cannot open wav list";


while(<WAVLIST>) {
  chomp;
  $sph = $_;
  @t = split("/",$sph);
  @t1 = split("[./]",$t[$#t]);
  $utt=$t1[0];
  if ($t1[1] eq "sph"){
  print WAV "$utt"," sph2pipe -f wav -p -c 1 $sph |\n";
  }elsif ($t1[1] eq "flac"){
  print WAV "${utt}", " sox -t flac $sph -r 8k -t wav - |\n";
  }else{
  print "no this wav format $t1[1]\n";
  }
  print SPKR "$utt $utt\n";
}
close(WAV) || die;
close(SPKR) || die;

while (<KEY>) {
  $line = $_;
  @toks = split(" ", $line);
  $spk = $toks[0];
  $uttsph = $toks[1];
  $utt = (split("[./]", $uttsph))[0];
  if ($utt ne "segmentid") {
    print TRIALS "${spk} ${utt}\n";
  }
}

close(TRIALS) || die;

if (system("utils/utt2spk_to_spk2utt.pl $out_dir_enroll/utt2spk >$out_dir_enroll/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir_enroll";
}
if (system("utils/utt2spk_to_spk2utt.pl $out_dir_test/utt2spk >$out_dir_test/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir_test";
}
if (system("utils/fix_data_dir.sh $out_dir_enroll") != 0) {
  die "Error fixing data dir $out_dir_enroll";
}
if (system("utils/fix_data_dir.sh $out_dir_test") != 0) {
  die "Error fixing data dir $out_dir_test";
}
