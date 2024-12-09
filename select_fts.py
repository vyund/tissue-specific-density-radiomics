import os

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
import shap
import pickle

from sklearn.metrics import roc_auc_score, confusion_matrix, RocCurveDisplay

from utils import *

'''
Script to select features from full (or 80%) of dataset after method validation from cross-val
'''

DATASET_NAME = 'i'
AGG_METHOD = 'median'
FT_SET = 'discrete'
XAI = True
NUM_FTS = 20
BINARY = True

if BINARY:
    TARGET_NAMES = ['Non-dense', 'Dense']
else:
    TARGET_NAMES = ['Density Grade A', 'Density Grade B', 'Density Grade C', 'Density Grade D']

RANDOM = False
if RANDOM:
    SEED = np.random.randint(0, 1e5)
else:
    SEED = 42

np.bool = np.bool_
np.int = np.int_

def load_data(data_path, label_path, exclude_path, label_type, prune=True, scale=True):
    data = pd.read_csv(data_path)

    with open(exclude_path) as f:
        if exclude_path == './ext_val/exclude.txt':
            exclude = f.readlines()
        else:
            exclude = f.read().splitlines()
    to_exclude = [e.split(' - ')[0] for e in exclude]
    
    data = data[~data['sample_name'].isin(to_exclude)]

    data = data.drop('Unnamed: 0', axis=1)

    labels = pd.read_csv(label_path)
    labels_filtered = select_label(labels, label=label_type)

    split_samples(data)
    data_agg = aggregate_views(data, aggregate_on='sample_id', method=AGG_METHOD)

    X = pd.merge(data_agg.astype({'sample_id': 'str'}), labels_filtered.astype({'DummyID': 'str'}).rename(columns={'DummyID': 'sample_id'}), on='sample_id')
    if BINARY:
        if label_type == 'Density_Overall':
            y = X[label_type].map({1: 0, 2: 0, 3: 1, 4: 1})
        else:
            y = X[label_type].map({0: 0, 1: 0, 2: 1, 3: 1})
    else:
        if label_type == 'Density_Overall':
            y = X[label_type] - 1
        else:
            y = X[label_type]
    ids = X['sample_id']

    X = X.drop(columns=['sample_id', label_type])

    all_fts = X.columns
 
    if scale:
        scaler = StandardScaler().fit(X)
        X[X.columns] = scaler.transform(X)
    else:
        scaler = 'No scaling performed'

    if prune:
        X, pruned_var = prune_var(X)
        X, pruned_corr = prune_corr(X)

    X = X.loc[:, (X != 0).any(axis=0)] # remove columns with all 0

    print(y.value_counts())
    num_classes = y.value_counts().shape[0]

    return X, y, num_classes, scaler, all_fts, ids

def plot_cm(cm, acc, plot_title, export_path):
    if BINARY:
        tick_labels = ['Non-dense', 'Dense']
    else:
        tick_labels = ['A', 'B', 'C', 'D']
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    combined = np.array([[f"{cm_val} ({cm_norm_val:.2f})" for cm_val, cm_norm_val in zip(cm_row, cm_norm_row)] for cm_row, cm_norm_row in zip(cm, cm_norm)])

    plt.figure()
    if XAI:
        sn.heatmap(cm, annot=combined, cbar=False, cmap='Blues', xticklabels=tick_labels, yticklabels=tick_labels, fmt='')
    else:
        sn.heatmap(cm, annot=combined, cbar=False, cmap='Reds', xticklabels=tick_labels, yticklabels=tick_labels, fmt='')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    if XAI:
        plt.title(plot_title)
        #plt.title('Validation Accuracy: {:.3f}'.format(acc))
    else:
        plt.title(plot_title)
        #plt.title('Validation Accuracy: {:.3f}'.format(acc))
    
    plt.savefig(export_path, dpi=400)
    plt.close()

def plot_roc(probs, y_test, target_names, plot_title, export_path):
    fig, ax = plt.subplots()
    auc_check = roc_auc_score(y_test, probs[:, 1])
    print('Binary AUC: {}'.format(auc_check))
    if auc_check < 0.5:
        print('AUC < 0.5')
        probs[:, 1] = 1 - probs[:, 1]
        auc_check_flipped = roc_auc_score(y_test, probs[:, 1])
        print('Binary AUC: {}'.format(auc_check_flipped))
    viz = RocCurveDisplay.from_predictions(
        y_test,
        probs[:, 1],
        name=f"ROC",
        lw=1,
        ax=ax
    )
    _ = ax.set(
        xlabel = 'False Positive Rate',
        ylabel = 'True Positive Rate',
        title = plot_title
    )
    plt.savefig(export_path, dpi=400)
    plt.close()

def plot_roc_multi(probs, y_onehot, num_classes, target_names, plot_title, export_path):
    fig, ax = plt.subplots()

    color_list = mpl.colormaps.get_cmap('Set2')
    colors = color_list.colors[0:num_classes]

    for c in range(num_classes):
        auc_check = roc_auc_score(y_onehot[:, c], probs[:, c])
        print('{} AUC: {}'.format(c, auc_check))
        if auc_check < 0.5:
            print('AUC < 0.5')
            probs[:, c] = 1 - probs[:, c]
            auc_check_flipped = roc_auc_score(y_onehot[:, c], probs[:, c])
            print('{} AUC: {}'.format(c, auc_check_flipped))
        viz = RocCurveDisplay.from_predictions(
            y_onehot[:, c],
            probs[:, c],
            name=f"ROC for {target_names[c]}",
            color=colors[c],
            lw=1,
            ax=ax
        )
    _ = ax.set(
        xlabel = 'False Positive Rate',
        ylabel = 'True Positive Rate',
        title = plot_title
    )
    plt.savefig(export_path, dpi=400)
    plt.close()

if __name__ == '__main__':
    if BINARY:
        #export_dir = './results/binary_results/select_fts'
        export_dir = './results/binary_ext_results'
    else:
        #export_dir = './selected_fts'
        export_dir = './results/ext_results'
    if DATASET_NAME == 'i':
        data_path = './extracted_fts/extracted_fts_{}.csv'.format(FT_SET)
        label_path = './labels/reports.csv'
        exclude_path = './exclude.txt'
        label_type = 'density_grade'

        data_path_val = './ext_val/extracted_fts_{}.csv'.format(FT_SET)
        label_path_val = './ext_val/reports.csv' 
        exclude_path_val = './ext_val/exclude.txt'
        label_type_val = 'Density_Overall'

    elif DATASET_NAME == 'ii':
        data_path = './ext_val/extracted_fts_{}.csv'.format(FT_SET)
        label_path = './ext_val/reports.csv'
        exclude_path = './ext_val/exclude.txt'
        label_type = 'Density_Overall'

        data_path_val = './extracted_fts/extracted_fts_{}.csv'.format(FT_SET)
        label_path_val = './labels/reports.csv'
        exclude_path_val = './exclude.txt'
        label_type_val = 'density_grade'
    
    X, y, num_classes, _, X_fts, _ = load_data(data_path, label_path, exclude_path, label_type, scale=False)
    X_val, y_val, num_classes_val, _, _, _ = load_data(data_path_val, label_path_val, exclude_path_val, label_type_val, prune=False, scale=False)

    # match features if !=
    X_val = X_val[X_val.columns.intersection(X.columns)]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    scaler = StandardScaler()
    X_train[X_train.columns] = scaler.fit_transform(X_train)
    X_test[X_test.columns] = scaler.transform(X_test)

    scaler_val = StandardScaler()
    X_val[X_val.columns] = scaler_val.fit_transform(X_val)

    clf = LogisticRegression(max_iter=500, random_state=SEED)

    print('Training Size: {}, Test Size: {}'.format(len(X_train), len(X_test)))

    # feature selection
    ft_selector = RFE(clf, n_features_to_select=NUM_FTS)
    ft_selector.fit(X_train, y_train)
    
    # rfe
    ft_ranks = ft_selector.ranking_
    ft_ranks_idx = enumerate(ft_ranks)
    sorted_ft_ranks_idx = sorted(ft_ranks_idx, key=lambda x: x[1])
    top_n_idx = [idx for idx, rnk in sorted_ft_ranks_idx[:NUM_FTS]]
    selected_fts = pd.Series(ft_selector.feature_names_in_[top_n_idx])

    X_train_rfe = X_train[selected_fts.to_list()]
    X_test_rfe = X_test[selected_fts.to_list()]
    clf.fit(X_train_rfe, y_train)

    if XAI:
        # shap
        max_evals = 2 * X_train_rfe.shape[1] + 1
        expl = shap.Explainer(clf.predict_proba, X_train_rfe, max_evals=max_evals)
        shap_values = expl(X_train_rfe)

        ft_names = X_train_rfe.columns
        all_selected_shap = []
        for c in range(num_classes):
            c_vals = np.abs(shap_values[:, :, c].values).mean(0)
            c_df = pd.DataFrame(list(zip(ft_names, c_vals)), columns=['ft_name', "ft_val"])
            c_df.sort_values(by=['ft_val'], ascending=False, inplace=True)
            c_shap_fts = c_df['ft_name'].head().to_list()
            all_selected_shap.extend(c_shap_fts)
        
        all_selected_df = pd.DataFrame(all_selected_shap)
        all_selected_df.to_csv(export_dir + '/{}_shap_selected_fts_{}.csv'.format(DATASET_NAME, SEED))
        selected_shap = list(set(all_selected_shap))

        # final fit and eval
        X_train_shap = X_train[selected_shap]
        X_test_shap = X_test[selected_shap]

        clf.fit(X_train_shap, y_train)

        ''' For SHAP plots if needed later
        # shap train
        max_evals_train = 2 * X_train_shap.shape[1] + 1
        explainer_train = shap.Explainer(clf.predict_proba, X_train_shap)
        #explainer_train = shap.Explainer(clf.predict_proba, X_train_shap, max_evals=max_evals_train)
        shap_values_train = explainer_train(X_train_shap)
        
        # shap test
        max_evals_test = 2 * X_test_shap.shape[1] + 1
        explainer_test = shap.Explainer(clf.predict_proba, X_test_shap)
        #explainer_test = shap.Explainer(clf.predict_proba, X_test_shap, max_evals=max_evals_test)
        shap_values_test = explainer_test(X_test_shap)
        '''

        preds = clf.predict(X_test_shap)
        acc = clf.score(X_test_shap, y_test)

        cm = confusion_matrix(y_test, preds)

        probs = clf.predict_proba(X_test_shap)

    else:
        all_selected_df = pd.DataFrame(selected_fts)
        all_selected_df.to_csv(export_dir + '/{}_selected_fts_{}.csv'.format(DATASET_NAME, SEED))

        preds = clf.predict(X_test_rfe)
        acc = clf.score(X_test_rfe, y_test)

        cm = confusion_matrix(y_test, preds)
        
        probs = clf.predict_proba(X_test_rfe)
    
    if XAI:
        ft_set = 'RFE-SHAP'
        ft_path = 'shap_'
    else:
        ft_set = 'RFE'
        ft_path = ''

    print('Testing...')
    y_onehot = label_binarize(y_test, classes=range(num_classes))

    if DATASET_NAME == 'i':
        if BINARY:
            roc_plot_title = '{} ROC: Trained and tested on I\n(Non-dense vs. Dense)'.format(ft_set)
            cm_plot_title = '{} CM: Trained and tested on I'.format(ft_set)
        else:
            roc_plot_title = '{} ROC: Trained and tested on I'.format(ft_set)
            cm_plot_title = '{} CM: Trained and tested on I'.format(ft_set)
    elif DATASET_NAME == 'ii':
        if BINARY:
            roc_plot_title = '{} ROC: Trained and tested on II\n(Non-dense vs. Dense)'.format(ft_set)
            cm_plot_title = '{} CM: Trained and tested on II'.format(ft_set)
        else:
            roc_plot_title = '{} ROC: Trained and tested on II'.format(ft_set)
            cm_plot_title = '{} CM: Trained and tested on II'.format(ft_set)
    
    roc_export_path = export_dir + '/{}{}_roc_{}.png'.format(ft_path, DATASET_NAME, SEED)
    cm_export_path = export_dir + '/{}{}_cm_{}.png'.format(ft_path, DATASET_NAME, SEED)

    if BINARY:
        plot_roc(probs, y_test, TARGET_NAMES, roc_plot_title, roc_export_path)
    else:
        plot_roc_multi(probs, y_onehot, num_classes, TARGET_NAMES, roc_plot_title, roc_export_path)
    plot_cm(cm, acc, cm_plot_title, cm_export_path)

    print('Validating...')
    if XAI:
        X_val = X_val[selected_shap]
    else:
        X_val = X_val[selected_fts]
    preds_val = clf.predict(X_val)
    acc_val = clf.score(X_val, y_val)

    cm_val = confusion_matrix(y_val, preds_val)

    probs_val = clf.predict_proba(X_val)

    y_onehot_val = label_binarize(y_val, classes=range(num_classes))

    if DATASET_NAME == 'i':
        if BINARY:
            roc_plot_title_val = '{} ROC: Trained on I, Validated on II\n(Non-dense vs. Dense)'.format(ft_set)
            cm_plot_title_val = '{} CM: Trained on I, Validated on II'.format(ft_set)
        else:
            roc_plot_title_val = '{} ROC: Trained on I, Validated on II'.format(ft_set)
            cm_plot_title_val = '{} CM: Trained on I, Validated on II'.format(ft_set)
    elif DATASET_NAME == 'ii':
        if BINARY:
            roc_plot_title_val = '{} ROC: Trained on II, Validated on I\n(Non-dense vs. Dense)'.format(ft_set)
            cm_plot_title_val = '{} CM: Trained on II, Validated on I'.format(ft_set)
        else:
            roc_plot_title_val = '{} ROC: Trained on II, Validated on I'.format(ft_set)
            cm_plot_title_val = '{} CM: Trained on II, Validated on I'.format(ft_set)

    roc_export_path_val = export_dir + '/{}{}_roc_val_{}.png'.format(ft_path, DATASET_NAME, SEED)
    cm_export_path_val = export_dir + '/{}{}_cm_val_{}.png'.format(ft_path, DATASET_NAME, SEED)

    if BINARY:
        plot_roc(probs_val, y_val, TARGET_NAMES, roc_plot_title_val, roc_export_path_val)
    else:
        plot_roc_multi(probs_val, y_onehot_val, num_classes_val, TARGET_NAMES, roc_plot_title_val, roc_export_path_val)
    
    plot_cm(cm_val, acc_val, cm_plot_title_val, cm_export_path_val)







    
