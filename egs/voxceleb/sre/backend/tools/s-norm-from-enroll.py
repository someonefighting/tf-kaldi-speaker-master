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

    parser.add_argument('--score-file', dest='score_file', type=str, help='score file')
    parser.add_argument('--snorm-file', dest='snorm_file', type=str, help='snorm file')
    parser.add_argument('--enroll-file', dest='enroll_file', type=str, help='enroll file')
    parser.add_argument('--eval-file', dest='eval_file', type=str, help='eval file')

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
    score_file = args.score_file
    snorm_file = args.snorm_file
    enroll_file = args.enroll_file
    eval_file = args.eval_file

    f_score = open(score_file, 'r')
    
    score_lines = f_score.readlines()

    score_lines = [line.strip() for line in score_lines]
    score_lines = [line.split('|') for line in score_lines if line != '']

    model_line = score_lines[0]
    scores = np.delete(score_lines, 0, axis=0)
    test_line = [var[0] for var in scores]
    scores = np.delete(scores, 0, axis=1)

    leng, wid = scores.shape

    scores = [float(score) for ss in scores for score in ss]
    scores = np.array(scores)
    scores.shape = leng, wid

    f_enroll = open(enroll_file, 'r')
    
    enroll_lines = f_enroll.readlines()
    enroll_lines = [line.strip().split(' ') for line in enroll_lines]
    mean_enroll = {}
    std_enroll = {}
    for enroll_line in enroll_lines:
        mean_enroll[enroll_line[0]] = float(enroll_line[1])
        std_enroll[enroll_line[0]] = float(enroll_line[2])
        
    f_eval = open(eval_file, 'r')
    
    eval_lines = f_eval.readlines()
    eval_lines = [line.strip().split(' ') for line in eval_lines]
    mean_test = {}
    std_test = {}
    for eval_line in eval_lines:
        mean_test[eval_line[0]] = float(eval_line[1])
        std_test[eval_line[0]] = float(eval_line[2])

    snorm_scores = np.zeros((leng, wid))

    means_l = np.zeros(wid)
    stds_l = np.zeros(wid)


    for ll in range(leng):
        for ww in range(wid):
            if scores[ll][ww] == -1000.0:
                snorm_scores[ll][ww] = scores[ll][ww]
            else:
                model_name = model_line[ww + 1]
                test_name = test_line[ll]
                snorm_scores[ll][ww] = (scores[ll][ww] - mean_test[test_name]) / std_test[test_name] + (scores[ll][ww] - mean_enroll[model_name]) / std_enroll[model_name]
                #(scores[ll][ww] - mean_ll) / std_ll t-norm
                #(scores[ll][ww] - mean_ww) / std_ww z-norm
                #(scores[ll][ww] - mean_ll) / std_ll + (scores[ll][ww] - mean_ww) / std_ww s-norm

    write_snorm(snorm_file, model_line, test_line, snorm_scores)
    
    f_score.close()
    f_enroll.close()
    f_eval.close()
            

def main():
    args = get_args()
    snorm(args)


if __name__ == "__main__":
    main()
