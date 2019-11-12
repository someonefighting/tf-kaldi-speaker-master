#!/usr/bin/perl
use warnings; #sed replacement for -w perl parameter
# Copyright 2019   Can Xu
# Apache 2.0
#
# Prepares NIST SRE16 enroll and test data in a single directory.
if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-SRE18-eval> <path-to-output>\n";
  print STDERR "e.g. $0 /export/corpora5/SRE/SRE2018/eval/ data/\n";
  exit(1);
}
($db_base, $out_dir) = @ARGV;

if (! -d "$db_base/data/") {
  print STDERR "Directory $db_base/data/ doesn't exist\n";
  exit(1);
}
$out_dir = "$out_dir/sre18";
$tmp_dir = "$out_dir/tmp";
if (system("mkdir -p $tmp_dir") != 0) {
  die "Error making directory $tmp_dir";
}

if (system("mkdir -p $out_dir") != 0) {
  print STDERR "Error making directory $out_dir\n";
  exit(1);
}

%seg2sph = ();
open(TRIALS, "<$db_base/docs/sre18_eval_trial_key.tsv") || die "Could not open $db_base/docs/sre16_eval_trial_key.tsv";
open(TRAIN, "<$db_base/docs/sre18_eval_enrollment.tsv") || die "Could not open $db_base/docs/sre16_eval_enrollment.tsv";
open(SPKR, ">$out_dir/utt2spk") || die "Could not open the output file $out_dir/utt2spk";
open(WAV, ">$out_dir/wav.scp") || die "Could not open the output file $out_dir/wav.scp";

if (system("find $db_base/data/ -name '*.sph' > $tmp_dir/sph.list") != 0) {
  die "Error getting list of sph files";
}
open(SPHLIST, "<$tmp_dir/sph.list") or die "cannot open wav list";
while(<SPHLIST>) {
  chomp;
  $sph = $_;
  @toks = split("/",$sph);
  $sph_id = (split("[./]",$toks[$#toks]))[0];
  $seg2sph{$sph_id} = $sph;
}

while (<TRAIN>) {
  chomp;
  $line = $_;
  @toks = split(" ", $line);
  $model = $toks[0];
  $seg = (split("[./]",$toks[1]))[0];
  $ch = $toks[2];
  if ($model ne "modelid") {
    if (exists $seg2sph{$seg}) {
      $sph = $seg2sph{$seg};
      if ($ch eq "a") {
        $utt = "${model}_${seg}_A";
        print WAV "$utt"," sph2pipe -f wav -p -c 1 $sph |\n";
        print SPKR "$utt ${model}\n";
      } elsif($ch eq "b") {
        $utt = "${model}_${seg}_B";
        print WAV "$utt"," sph2pipe -f wav -p -c 2 $sph |\n";
        print SPKR "$utt ${model}\n";
      } else {
        print STDERR "Malformed train file\n";
        exit(1);
      }
    }
  }
}

while (<TRIALS>) {
  chomp;
  $line = $_;
  @toks = split(" ", $line);
  $model = $toks[0];
  $seg = (split("[./]",$toks[1]))[0];
  $ch = $toks[2];
  $target = $toks[3];
  if (exists $seg2sph{$seg} and -f $seg2sph{$seg}) {
    $sph = $seg2sph{$seg};
    if ($target eq "target") {
      if ($ch eq "a") {
        $utt = "${model}_${seg}_A";
        print WAV "$utt"," sph2pipe -f wav -p -c 1 $sph |\n";
        print SPKR "$utt $model\n";
      } elsif($ch eq "b") {
        $utt = "${model}_${seg}_B";
        print WAV "$utt"," sph2pipe -f wav -p -c 2 $sph |\n";
        print SPKR "$utt $model\n";
      } else {
        print STDERR "Malformed trials file\n";
        exit(1);
      }
    }
  }
}

close(TRIALS) || die;
close(TRAIN) || die;
close(SPKR) || die;
close(WAV) || die;

if (system(
  "utils/utt2spk_to_spk2utt.pl $out_dir/utt2spk >$out_dir/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir";
}

if (system("utils/fix_data_dir.sh $out_dir")) {
  die "Error validating directory $out_dir";
}

