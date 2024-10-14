import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

import shap
import pickle

from utils import *

'''
Script to run nested 5-fold cross-validation for classification performance
'''

LABEL = 'density_grade'
AGG_METHOD = 'median'

METRIC = 'accuracy'
NUM_FTS = 20
FT_CLASSES = {'shape2D': [], 'firstorder': [], 'glcm': [], 'glrlm': [], 'glszm': [], 'gldm': []}

SAVE_FIG = False
NUM_FTS_VIEW = 50

FT_SET = 'dense'
PLOT_TYPE = 'bar'

TRAIN = True

with open('exclude.txt') as f:
    exclude = f.read().splitlines()
EXCLUDE = [e.split(' - ')[0] for e in exclude]

np.bool = np.bool_
np.int = np.int_

def calc_metrics(clf, X_test, y_train, y_test, num_classes):
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)
    acc = clf.score(X_test, y_test)

    cm = confusion_matrix(y_test, preds)
    cr = classification_report(y_test, preds)

    corr_matrix = np.corrcoef(preds, y_test)
    pearson_r = corr_matrix[0, 1]

    aucs = []
    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)
    for c in range(num_classes):
        try:
            c_auc = roc_auc_score(y_onehot_test[:, c], probs[:, c], average='weighted')
        except ValueError:
            c_auc = -1.0
        aucs.append(c_auc)
    
    return acc, aucs, cm, cr, pearson_r

if __name__ == '__main__':
    data_path = './extracted_fts/extracted_fts_{}.csv'.format(FT_SET)
    label_path = './labels/reports.csv'
    
    data = pd.read_csv(data_path)
    data = data[~data['sample_name'].isin(EXCLUDE)]
    data = data.drop('Unnamed: 0', axis=1)

    labels = pd.read_csv(label_path)
    labels_filtered = select_label(labels, label=LABEL)

    split_samples(data)
    data_agg = aggregate_views(data, aggregate_on='sample_id', method=AGG_METHOD)

    X = pd.merge(data_agg.astype({'sample_id': 'str'}), labels_filtered.astype({'DummyID': 'str'}).rename(columns={'DummyID': 'sample_id'}), on='sample_id')
    y = X[LABEL]
    ids = X['sample_id']

    X = X.drop(columns=['sample_id', LABEL])

    X, pruned_var = prune_var(X)
    X, pruned = prune_corr(X)
    X[X.columns] = StandardScaler().fit_transform(X)

    X = X.loc[:, (X != 0).any(axis=0)] # remove columns with all 0

    print(y.value_counts())
    num_classes = y.value_counts().shape[0]

    outer_cv = StratifiedKFold(5, shuffle=True, random_state=123)
    inner_cv = StratifiedKFold(5, shuffle=True, random_state=321)

    # cols = [outer_i, best_inner_i, acc, aucs, cm, cr, selected_fts]
    results = []

    for outer_i, (train_outer, test_outer) in enumerate(outer_cv.split(X, y)):
        X_train_outer, X_test_outer = X.iloc[train_outer], X.iloc[test_outer]
        y_train_outer, y_test_outer = y.iloc[train_outer], y.iloc[test_outer]

        # cols = [outer_i, inner_i, acc, aucs, cm, cr, selected_fts]
        inner_results = []
        for inner_i, (train_inner, test_inner) in enumerate(inner_cv.split(X_train_outer, y_train_outer)):
            X_train_inner, X_test_inner = X_train_outer.iloc[train_inner], X_train_outer.iloc[test_inner]
            y_train_inner, y_test_inner = y_train_outer.iloc[train_inner], y_train_outer.iloc[test_inner]

            clf = LogisticRegression(max_iter=500, random_state=123)
            ft_selector = RFE(clf, n_features_to_select=NUM_FTS)
            ft_selector.fit(X_train_inner, y_train_inner)

            ft_ranks = ft_selector.ranking_
            ft_ranks_idx = enumerate(ft_ranks)
            sorted_ft_ranks_idx = sorted(ft_ranks_idx, key=lambda x: x[1])
            top_n_idx = [idx for idx, rnk in sorted_ft_ranks_idx[:NUM_FTS]]
            selected_fts = pd.Series(ft_selector.feature_names_in_[top_n_idx])

            X_train_inner_rfe = X_train_inner[selected_fts.to_list()]
            X_test_inner_rfe = X_test_inner[selected_fts.to_list()]

            clf.fit(X_train_inner_rfe, y_train_inner)
            
            # shap
            max_evals = 2 * X_train_inner_rfe.shape[1] + 1
            explainer = shap.Explainer(clf.predict_proba, X_train_inner_rfe, max_evals=max_evals)
            shap_values = explainer(X_train_inner_rfe)

            ft_names = X_train_inner_rfe.columns
            all_selected_shap = []
            for c in range(num_classes):
                c_vals = np.abs(shap_values[:, :, c].values).mean(0)
                c_df = pd.DataFrame(list(zip(ft_names, c_vals)), columns=['ft_name', "ft_val"])
                c_df.sort_values(by=['ft_val'], ascending=False, inplace=True)
                c_shap_fts = c_df['ft_name'].head().to_list()
                all_selected_shap.extend(c_shap_fts)

            selected_shap = list(set(all_selected_shap))

            X_train_inner_shap = X_train_inner[selected_shap]
            X_test_inner_shap = X_test_inner[selected_shap]

            clf.fit(X_train_inner_shap, y_train_inner)

            # metrics
            acc, aucs, cm, cr, _ = calc_metrics(clf, X_test_inner_shap, y_train_inner, y_test_inner, num_classes)
            inner_results.append([outer_i, inner_i, acc, aucs, cm, cr, selected_shap])
        
        inner_df = pd.DataFrame(inner_results, columns=['outer_i', 'inner_i', 'acc', 'auc', 'cm', 'cr', 'fts'])
        best_inner_i = inner_df['acc'].idxmax()
        best_inner_fts = inner_df.iloc[best_inner_i]['fts']

        X_train_outer_best = X_train_outer[best_inner_fts]
        X_test_outer_best = X_test_outer[best_inner_fts]

        clf = LogisticRegression(max_iter=500, random_state=123)
        clf.fit(X_train_outer_best, y_train_outer)

        preds = clf.predict(X_test_outer_best)
        test = X_test_outer_best.copy()
        test['preds'] = preds
        test['labels'] = y_test_outer
        test = pd.merge(test, ids, left_index=True, right_index=True)
        misclassified = test.loc[~(test['preds'] == test['labels'])]
        #misclassified.to_csv('./misclassified/outer_{}.csv'.format(outer_i))

        acc, aucs, cm, cr, pearson_r = calc_metrics(clf, X_test_outer_best, y_train_outer, y_test_outer, num_classes)
        results.append([outer_i, best_inner_i, pearson_r, acc, aucs, cm, cr, best_inner_fts])
        print(2)
    
    results_df = pd.DataFrame(results, columns=['outer_i', 'best_inner_i', 'pearson_r', 'acc', 'auc', 'cm', 'cr', 'fts'])
    results_df.to_csv('./results/nested_cv_results/nested_cv_results.csv')
    print(3)

