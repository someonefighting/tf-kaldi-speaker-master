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
    parser.add_argument('--enroll-file', dest='enroll_file', type=str, help='score file')

    args = parser.parse_args()

    return args

def write_snorm(enroll_file, model_line, means_l, stds_l):
    f_snorm = open(enroll_file, 'w')
    
    len_model = len(model_line)
    
    
    for index in range(len_model):
        f_snorm.write(model_line[index] + ' ')
        f_snorm.write(str(means_l[index]))
        f_snorm.write(' ')
        f_snorm.write(str(stds_l[index]))
        f_snorm.write('\n')
    
    f_snorm.close()


def snorm(args):
    score_file = args.score_file
    enroll_file = args.enroll_file

    f_score = open(score_file, 'r')
    score_lines = f_score.readlines()

    score_lines = [line.strip() for line in score_lines]
    score_lines = [line.split('|') for line in score_lines if line != '']

    model_line = score_lines[0].copy()
    del model_line[0]
    scores = np.delete(score_lines, 0, axis=0)
    test_line = [var[0] for var in scores]
    scores = np.delete(scores, 0, axis=1)

    leng, wid = scores.shape

    scores = [float(score) for ss in scores for score in ss]
    scores = np.array(scores)
    scores.shape = leng, wid

    

    snorm_scores = np.zeros((leng, wid))

    means_w = np.zeros(wid)
    stds_w = np.zeros(wid)


    for ww in range(wid):
        score_ww = scores[:,ww].copy()
        score_ww.sort()
        for i in range(leng):
            if score_ww[i] != -1000.0:
                break
        score_ww = score_ww[-int(leng*0.3):]
        #print(ww)
        means_w[ww] = np.mean(score_ww)
        stds_w[ww]  = np.std(score_ww, ddof=1)
        del score_ww

    write_snorm(enroll_file, model_line, means_w, stds_w)
    
    f_score.close()
            

def main():
    args = get_args()
    snorm(args)


if __name__ == "__main__":
    main()
