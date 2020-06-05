import os
import numpy as np
import logging
import argparse
import sys

logger = logging.getLogger('s-norm score.')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Starting s-norm')

def get_args():
    """
    get args from stdin.
    """

    parser = argparse.ArgumentParser(description='snorm score.', formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    conflict_handler='resolve')

    parser.add_argument('--matrix-file', dest='matrix_file', type=str, help='matrix_file file')
    parser.add_argument('--trial-file', dest='trial_file', type=str, help='trial_file file')
    parser.add_argument('--kaldi-file', dest='kaldi_file', type=str, help='kaldi_file file')

    args = parser.parse_args()

    return args

def write_snorm(snorm_file, model_line, test_line, snorm_scores):
    f_snorm = open(snorm_file, 'w')
    for model in model_line:
        if model == model_line[-1]:
            f_snorm.write(model)
            f_snorm.write('\n')
        else:
            f_snorm.write(model)
            f_snorm.write('|')
    
    line = 0
    for test in test_line:
        f_snorm.write(test)
        f_snorm.write('|')
        index = 0
        for score in snorm_scores[line]:
            if score == snorm_scores[line][-1]:
                f_snorm.write("%.6f"%snorm_scores[line][index])
                f_snorm.write('\n')
            else:
                f_snorm.write("%.6f"%snorm_scores[line][index])
                f_snorm.write('|')
            index = index + 1
        line = line + 1
    f_snorm.close()


def snorm(args):
    matrix_file = args.matrix_file
    trial_file = args.trial_file
    kaldi_file = args.kaldi_file


    f_matrix = open(matrix_file, 'r')
    
    score_lines = f_matrix.readlines()

    score_lines = [line.strip() for line in score_lines]
    score_lines = [line.split('|') for line in score_lines if line != '']

    model_line = score_lines[0]
    #del model_line[0]
    scores = np.delete(score_lines, 0, axis=0)
    test_line = [var[0] for var in scores]
    scores = np.delete(scores, 0, axis=1)

    leng, wid = scores.shape

    scores = [score for ss in scores for score in ss]
    scores = np.array(scores)
    scores.shape = leng, wid

    f_trial = open(trial_file, 'r')
    
    enroll_lines = f_trial.readlines()
    enroll_lines = [line.strip().split('\t') for line in enroll_lines]
    enroll_lines = np.delete(enroll_lines, 0, axis=0)
    
    f_kaldi = open(kaldi_file, "w")
    f_kaldi.write("modelid\tsegmentid\tside\tLLR\n")
    
    for line in enroll_lines:
        model_index = model_line.index(line[0])-1
        test_index  = test_line.index(line[1].split('.')[0])
        f_kaldi.write(line[0]+"\t"+line[1]+"\t"+"a\t"+scores[test_index][model_index]+"\n")
    
    
    #write_snorm(snorm_file, model_line, test_line, snorm_scores)
    
    f_matrix.close()
    f_trial.close()
    f_kaldi.close()
            

def main():
    args = get_args()
    snorm(args)


if __name__ == "__main__":
    main()
