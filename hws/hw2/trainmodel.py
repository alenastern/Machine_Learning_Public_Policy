from __future__ import division
import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt
from scipy import optimize
import time
import seaborn as sns
import sys
sys.path.append('/Users/alenastern/Documents/Spring2018/Machine_Learning/Machine_Learning_Public_Policy/hws/hw2')
import preprocess as pp

def temporal_validate(start_time, end_time, prediction_windows):
'''
Starting from start time, create training sets incrementing in number of months specified by prediction_window, with 
test set beginning one day following the end of training set for a duration of the number of months specified by
prediction_window. Continue until end_time is reached.

Returns list outlining train start, train end, test start, and test end for all temporal splits.
'''

    from datetime import date, datetime, timedelta
    from dateutil.relativedelta import relativedelta

    start_time_date = datetime.strptime(start_time, '%Y-%m-%d')
    end_time_date = datetime.strptime(end_time, '%Y-%m-%d')

    for prediction_window in prediction_windows:
        windows = 1
        test_end_time = start_time_date
        while (end_time_date >= test_end_time + relativedelta(months=+prediction_window)):
            train_start_time = start_time_date
            train_end_time = train_start_time + windows * relativedelta(months=+prediction_window) - relativedelta(days=+1)
            test_start_time = train_end_time + relativedelta(days=+1)
            test_end_time = test_start_time  + relativedelta(months=+prediction_window) - relativedelta(days=+1)
            temp_split.append([train_start_time,train_end_time,test_start_time,test_end_time,prediction_window])


            windows += 1

    return temp_split
            
     

def joint_sort_descending(l1, l2):
    '''
    Sorts y_test and y_pred in descending order of probability.
    Adapted with permission from Rayid Ghani, Data Science for Social Good: https://github.com/rayidghani/magicloops 
    '''
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]

def generate_binary_at_k(y_scores, k):
    '''
    Converts probability score into binary outcome measure based upon cutoff.
    Adapted with permission from Rayid Ghani, Data Science for Social Good: https://github.com/rayidghani/magicloops 
    '''
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary

def precision_at_k(y_true, y_scores, k):
    '''
    Calculates precision at given threshold k.
    Adapted with permission from Rayid Ghani, Data Science for Social Good: https://github.com/rayidghani/magicloops 
    '''
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)
    precision = precision_score(y_true, preds_at_k)
    return precision

def recall_at_k(y_true, y_scores, k):
    '''
    Calculates recall at given threshold k.
    '''
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    recall_at_k = generate_binary_at_k(y_scores, k)
    recall = recall_score(y_true, recall_at_k)
    return recall

def plot_precision_recall_n(y_true, y_prob, model_name):
    '''
    Plots precision-recall curve for a given model.
    Adapted with permission from Rayid Ghani, Data Science for Social Good: https://github.com/rayidghani/magicloops 
    '''
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    name = model_name
    plt.title(name)
    #plt.savefig(name)
    plt.show()

    
def temporal_split(total_data, train_start, train_end, test_start, test_end, time_var, pred_var):
    '''
    Splits data into train set and test set given temporal split train start/end and test start/end times.
    
    Returns X train/test sets and Y train/test sets.
    '''
    train_data = total_data[(total_data[time_var] >= train_start) & (total_data[time_var] <= train_end)]
    train_data.drop([time_var], axis = 1)
    y_train = train_data[pred_var]
    X_train = train_data.drop([pred_var, time_var], axis = 1)

    test_data = total_data[(total_data[time_var] >= test_start) & (total_data[time_var] <= test_end)]
    test_data.drop([time_var], axis = 1)
    y_test = test_data[pred_var]
    X_test = test_data.drop([pred_var, time_var], axis = 1)

    return X_train, X_test, y_train, y_test



def run_models(models_to_run, classifiers, parameters, total_data, pred_var, temporal_validate = None, time_var= None):
    """
    Given a set of models to run and parameters, runs each model for each combination of parameters. If a set of temporal 
    splits is provided, also runs each set of models/parameters for each temporal split in the model. 
    
    Returns table with temporal split, model, and parameter information along with model performance on a number of specified
    metrics including auc-roc, precision at different levels, recall at different levels, and F1 at different levels.
    
    Table also includes baseline of the prevalence of the positive outcome label in the population for each temporal split. 
    """
    results_df =  pd.DataFrame(columns=('train_start', 'train_end', 'test_start', 'test_end', 'model_type','clf', 'parameters', 'auc-roc',
        'p_at_1', 'p_at_2', 'p_at_5', 'p_at_10', 'p_at_20', 'p_at_30', 'p_at_50', 'r_at_1', 'r_at_2', 'r_at_5', 'r_at_10', 'r_at_20', 'r_at_30', 'r_at_50',
        'f1_at_2', 'f1_at_20', 'f1_at_50'))
    if temporal_validate:
        for t in temporal_validate:
            train_start, train_end, test_start, test_end = t[0], t[1], t[2], t[3]
            X_train, X_test, y_train, y_test = temporal_split(total_data, train_start, train_end, test_start, test_end, time_var, pred_var)
            X_train.students_reached_bins = X_train.students_reached_bins.fillna(X_train.students_reached_bins.median()) 
            X_test.students_reached_bins = X_test.students_reached_bins.fillna(X_test.students_reached_bins.median()) 
            for index, classifier in enumerate([classifiers[x] for x in models_to_run]):
                print("Running through model {}...".format(models_to_run[index]))
                parameter_values = parameters[models_to_run[index]]
                for p in ParameterGrid(parameter_values):
                    print(p)
                    try:
                        classifier.set_params(**p)
                        y_pred_probs = classifier.fit(X_train, y_train).predict_proba(X_test)[:,1]
                        y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                        p_at_2 = precision_at_k(y_test_sorted,y_pred_probs_sorted,2.0)
                        r_at_2 = recall_at_k(y_test_sorted,y_pred_probs_sorted,2.0)
                        p_at_20 = precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0)
                        r_at_20 = recall_at_k(y_test_sorted,y_pred_probs_sorted,20.0)
                        p_at_50 = precision_at_k(y_test_sorted,y_pred_probs_sorted,50.0)
                        r_at_50 = recall_at_k(y_test_sorted,y_pred_probs_sorted,50.0)
                        results_df.loc[len(results_df)] = [train_start, train_end, test_start, test_end,
                                                           models_to_run[index],classifier, p,
                                                           roc_auc_score(y_test_sorted, y_pred_probs),
                                                           precision_at_k(y_test_sorted,y_pred_probs_sorted,1.0),
                                                           p_at_2,
                                                           precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                           precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                           p_at_20,
                                                           precision_at_k(y_test_sorted,y_pred_probs_sorted,30.0),
                                                           p_at_50,
                                                           recall_at_k(y_test_sorted,y_pred_probs_sorted,1.0),
                                                           r_at_2,
                                                           recall_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                           recall_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                           r_at_20,
                                                           recall_at_k(y_test_sorted,y_pred_probs_sorted,30.0),
                                                           r_at_50,
                                                           2*(p_at_2*r_at_2)/(p_at_2 + r_at_2),
                                                           2*(p_at_20*r_at_20)/(p_at_20 + r_at_20),
                                                           2*(p_at_50*r_at_50)/(p_at_50 + r_at_50)]
                    except IndexError as e:
                        print('Error:',e)
                        continue
            results_df.loc[len(results_df)] = [train_start, train_end, test_start, test_end, "baseline", '',
                        '', y_test.sum()/len(y_test), '', '', '', '', '', '', '', '', '', '', '', '', '','', '', '', '']
            
        return results_df

