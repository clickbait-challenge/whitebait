#!/usr/bin/python3
'''evaluation program for the clickbait challenge 2017.
web:
    http://www.clickbait-challenge.org/
authors:
    tim.gollub@uni-weimar.de
    matti.wiegmann@uni-weimar.de
'''

import json
import sys
import sklearn.metrics as skm
import numpy as np

UNDERLINE = '\033[4m'
END = '\033[0m'


def usage():
    usage = ''' Usage:
    ~$ python eval.py "truth.jsonl" "predictions.jsonl"
    '''
    print(usage)
    exit()


def write_result(key, value):
    value = round(value, ndigits=3)##Added by phil
    print(key + ': ' + str(value))


def normalized_mean_squared_error(truth, predictions):
    norm = skm.mean_squared_error(truth, np.full(len(truth), np.mean(truth)))
    return skm.mean_squared_error(truth, predictions) / norm


regression_measures = {'Explained variance': skm.explained_variance_score,
                       'Mean absolute error': skm.mean_absolute_error,
                       'Mean squared error': skm.mean_squared_error,
                       'Median absolute error': skm.median_absolute_error,
                       'R2 score': skm.r2_score,
                       'Normalized mean squared error': normalized_mean_squared_error}

classification_measures = {'Accuracy': skm.accuracy_score,
                           'Precision': skm.precision_score,
                           'Recall': skm.recall_score,
                           'F1 score': skm.f1_score}

if __name__ == "__main__":
    print("Start")
    try:
        with open(sys.argv[1], "r") as truth_file:
            truth_dict = {json.loads(s)['id']: json.loads(s)['truthMean']
                          for s in truth_file.readlines()}

        with open(sys.argv[1], "r") as truth_file:
            class_dict = {json.loads(s)['id']: json.loads(s)['truthClass']
                          for s in truth_file.readlines()}

        with open(sys.argv[2], "r") as preditcions_file:
            predictions_dict = {json.loads(s)['id']: json.loads(s)['clickbaitScore']
                                for s in preditcions_file.readlines()}
    except (KeyError, IndexError):
        usage()

    try:
        truth = []
        classes = []
        predictions = []
        for key in truth_dict:
            truth.append(truth_dict[key])
            classes.append(class_dict[key])
            predictions.append(predictions_dict[key])
    except KeyError:
        print('missing id in predictions.')
        exit()

    print(UNDERLINE + '\nDataset Stats' + END)
    write_result('Size', len(truth))
    sum_clickbait = sum(1 for x in classes if x == 'clickbait')
    write_result('#Clickbait', sum_clickbait)
    write_result('#No-Clickbait', len(truth) - sum_clickbait)

    print(UNDERLINE + '\nRegression scores' + END)
    for name in sorted(regression_measures):
        write_result(name,
                     regression_measures[name](truth, predictions)
                )

    print(UNDERLINE + '\nBinary classification scores' + END)
    classes = [0 if t == 'no-clickbait' else 1 for t in classes]
    predictions = [0 if t < 0.5 else 1 for t in predictions]
    for name in sorted(classification_measures):
        write_result(name,
                     classification_measures[name](classes, predictions)
                     )

#    print(UNDERLINE + '\nClassification report' + END)
#    print(skm.classification_report(classes, predictions))
