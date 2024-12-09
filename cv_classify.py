import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.preprocessing import StandardScaler, LabelBinarizer, label_binarize
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression
import shap
import pickle

from sklearn.metrics import roc_auc_score, confusion_matrix, RocCurveDisplay, auc

from utils import *

'''
Script to run standard cross-validation and generate pooled confusion matrix or per-class ROC curves
'''

AGG_METHOD = 'median'
FT_SET = 'discrete'
LABEL = 'density_grade'
XAI = True
BINARY = True
POOLED = True
ROC = True
CM = True
NUM_FTS = 20

RANDOM = False
if RANDOM:
    SEED = np.random.randint(0, 1e5)
else:
    SEED = 42

with open('exclude.txt') as f:
    exclude = f.read().splitlines()
EXCLUDE = [e.split(' - ')[0] for e in exclude]

np.bool = np.bool_
np.int = np.int_

if __name__ == '__main__':
    if BINARY:
        export_dir = './results/binary_results'
    else:
        export_dir = './results/cv_results'
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
    if BINARY:
        y = X[LABEL].map({0: 0, 1: 0, 2: 1, 3: 1})
    else:
        y = X[LABEL]
    ids = X['sample_id']

    X = X.drop(columns=['sample_id', LABEL])

    X, pruned_var = prune_var(X)
    X, pruned = prune_corr(X)

    X = X.loc[:, (X != 0).any(axis=0)] # remove columns with all 0

    num_classes = y.value_counts().shape[0]

    n_size = int(len(X) * .8)

    accs = []
    aucs = [[] for i in range(num_classes)]
    tprs = [[] for i in range(num_classes)]
    cms = np.zeros((num_classes, num_classes), dtype=int)

    num_splits = 5
    cv = StratifiedKFold(num_splits, shuffle=True, random_state=SEED)
    
    if BINARY:
        fig, ax = plt.subplots()
        target_names = ['Non-dense', 'Dense']
    else:
        # make figs for each class
        fig0, ax0 = plt.subplots()
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
        
        select_class = [ax0, ax1, ax2, ax3]
        figs = [fig0, fig1, fig2, fig3]
        target_names = ['Density Grade A', 'Density Grade B', 'Density Grade C', 'Density Grade D']
    mean_fpr = np.linspace(0, 1, 100)

    pooled_preds = []
    pooled_probs = []
    pooled_labels = []
    
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        clf = LogisticRegression(max_iter=500, random_state=SEED)

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_train[X_train.columns] = scaler.fit_transform(X_train)
        X_test[X_test.columns] = scaler.transform(X_test)

        if i == 0:
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
            #all_selected_df.to_csv('./shap_cv/selected_fts_{}.csv'.format(i))
            selected_shap = list(set(all_selected_shap))

            # final fit and eval
            X_train_shap = X_train[selected_shap]
            X_test_shap = X_test[selected_shap]

            clf.fit(X_train_shap, y_train)

            # shap train
            max_evals_train = 2 * X_train_shap.shape[1] + 1
            explainer_train = shap.Explainer(clf.predict_proba, X_train_shap)
            #explainer_train = shap.Explainer(clf.predict_proba, X_train_shap, max_evals=max_evals_train)
            shap_values_train = explainer_train(X_train_shap)

            #with open('./shap_cv/shap_train_{}.pkl'.format(i), 'wb') as outp:
            #    pickle.dump(shap_values_train, outp, pickle.HIGHEST_PROTOCOL)
            
            # shap test
            max_evals_test = 2 * X_test_shap.shape[1] + 1
            explainer_test = shap.Explainer(clf.predict_proba, X_test_shap)
            #explainer_test = shap.Explainer(clf.predict_proba, X_test_shap, max_evals=max_evals_test)
            shap_values_test = explainer_test(X_test_shap)

            #with open('./shap_cv/shap_test_{}.pkl'.format(i), 'wb') as outp:
            #    pickle.dump(shap_values_test, outp, pickle.HIGHEST_PROTOCOL)

            preds = clf.predict(X_test_shap)
            acc = clf.score(X_test_shap, y_test)
            accs.append(acc)

            cm = confusion_matrix(y_test, preds)
            cms = cms + cm
            
            probs = clf.predict_proba(X_test_shap)

        else:
            preds = clf.predict(X_test_rfe)
            acc = clf.score(X_test_rfe, y_test)
            accs.append(acc)

            cm = confusion_matrix(y_test, preds)
            cms = cms + cm
            
            probs = clf.predict_proba(X_test_rfe)
        
        if ROC:
            if BINARY:
                auc_check = roc_auc_score(y_test, probs[:, 1])
                if auc_check < 0.5:
                    probs[:, 1] = 1 - probs[:, 1]
                viz = RocCurveDisplay.from_predictions(
                        y_test,
                        probs[:, 1],
                        name=f"ROC - Fold {i+1}",
                        alpha=0.3,
                        lw=1,
                        ax=ax,
                )
                interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0
                tprs[1].append(interp_tpr)
                aucs[1].append(viz.roc_auc)
            else:
                label_binarizer = LabelBinarizer().fit(y_train)
                y_onehot_test = label_binarizer.transform(y_test)

                for c in range(num_classes):
                    auc_check = roc_auc_score(y_onehot_test[:, c], probs[:, c])
                    if auc_check < 0.5:
                        probs[:, c] = 1 - probs[:, c]
                    viz = RocCurveDisplay.from_predictions(
                        y_onehot_test[:, c],
                        probs[:, c],
                        name=f"ROC - Fold {i+1}",
                        alpha=0.3,
                        lw=1,
                        ax=select_class[c],
                    )
                    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                    interp_tpr[0] = 0.0
                    tprs[c].append(interp_tpr)
                    aucs[c].append(viz.roc_auc)
        
        pooled_preds.append(preds)
        pooled_probs.append(probs)
        pooled_labels.append(y_test)

    if XAI:
        mean_color = 'b'
    else:
        mean_color = 'r'
    
    pooled_preds_flat = [x for xs in pooled_preds for x in xs]
    pooled_probs_flat = [x for xs in pooled_probs for x in xs]
    pooled_labels_flat = [x for xs in pooled_labels for x in xs]
    pooled_df = pd.DataFrame(pooled_probs_flat)
    pooled_onehot = label_binarize(pooled_labels_flat, classes=range(num_classes))

    pooled_df['label'] = pooled_labels_flat
    
    if XAI:
        pooled_df.to_csv(export_dir + '/shap_pooled_probs_{}.csv'.format(SEED))
    else:
        pooled_df.to_csv(export_dir + '/pooled_probs_{}.csv'.format(SEED))
    
    # roc curves
    if ROC:
        if BINARY:
            mean_tpr = np.mean(tprs[1], axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs[1])
            print('Binary AUC: {} +- {}'.format(mean_auc, std_auc))
            if POOLED:
                auc_check = roc_auc_score(pooled_labels_flat, pooled_df[1])
                if auc_check < 0.5:
                    pooled_df[1] = 1 - pooled_df[1]
                viz = RocCurveDisplay.from_predictions(
                    pooled_labels_flat,
                    pooled_df[1],
                    color=mean_color,
                    name=f"Pooled ROC",
                    lw=2,
                    alpha=0.8,
                    ax=ax
                )
            else:
                ax.plot(
                    mean_fpr,
                    mean_tpr,
                    color=mean_color,
                    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
                    lw=2,
                    alpha=0.8,
                )
            
            lims = [
                    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
            ]
            ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)

            std_tpr = np.std(tprs[1], axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(
                mean_fpr,
                tprs_lower,
                tprs_upper,
                color="grey",
                alpha=0.2,
                label=r"$\pm$ 1 std. dev.",
            )
            if XAI:
                ax.set(
                    xlabel="False Positive Rate",
                    ylabel="True Positive Rate",
                    title=f"RFE-SHAP Cross-validated ROC \n(Non-dense vs. Dense)",
                )
            else:
                ax.set(
                    xlabel="False Positive Rate",
                    ylabel="True Positive Rate",
                    title=f"RFE Cross-validated ROC \n(Non-dense vs. Dense)",
                )
            ax.legend(loc="lower right")
            if XAI:
                fig.savefig(export_dir + '/shap_roc_{}.png'.format(SEED), dpi=400)
            else:
                fig.savefig(export_dir + '/roc_{}.png'.format(SEED), dpi=400)
        else:
            for c in range(num_classes):
                mean_tpr = np.mean(tprs[c], axis=0)
                mean_tpr[-1] = 1.0
                mean_auc = auc(mean_fpr, mean_tpr)
                std_auc = np.std(aucs[c])
                print('{} AUC: {} +- {}'.format(c, mean_auc, std_auc))
                if POOLED:
                    auc_check = roc_auc_score(pooled_onehot[:, c], pooled_df[c])
                    if auc_check < 0.5:
                        pooled_df[c] = 1 - pooled_df[c]
                    viz = RocCurveDisplay.from_predictions(
                        pooled_onehot[:, c],
                        pooled_df[c],
                        color=mean_color,
                        name=f"Pooled ROC",
                        lw=2,
                        alpha=0.8,
                        ax=select_class[c]
                    )
                else:
                    select_class[c].plot(
                        mean_fpr,
                        mean_tpr,
                        color=mean_color,
                        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
                        lw=2,
                        alpha=0.8,
                    )

                lims = [
                    np.min([select_class[c].get_xlim(), select_class[c].get_ylim()]),  # min of both axes
                    np.max([select_class[c].get_xlim(), select_class[c].get_ylim()]),  # max of both axes
                ]
                select_class[c].plot(lims, lims, 'k--', alpha=0.75, zorder=0)

                std_tpr = np.std(tprs[c], axis=0)
                tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
                select_class[c].fill_between(
                    mean_fpr,
                    tprs_lower,
                    tprs_upper,
                    color="grey",
                    alpha=0.2,
                    label=r"$\pm$ 1 std. dev.",
                )
                if XAI:
                    select_class[c].set(
                        xlabel="False Positive Rate",
                        ylabel="True Positive Rate",
                        title=f"RFE-SHAP Cross-validated ROC \n(Positive label '{target_names[c]}')",
                    )
                else:
                    select_class[c].set(
                        xlabel="False Positive Rate",
                        ylabel="True Positive Rate",
                        title=f"RFE Cross-validated ROC \n(Positive label '{target_names[c]}')",
                    )
                select_class[c].legend(loc="lower right")
            for i, fig in enumerate(figs):
                if XAI:
                    fig.savefig(export_dir + '/shap_roc_{}_{}.png'.format(i, SEED), dpi=400)
                else:
                    fig.savefig(export_dir + '/roc_{}_{}.png'.format(i, SEED), dpi=400)

    # confusion matrix
    if CM:
        if BINARY:
            tick_labels = ['Non-dense', 'Dense']
        else:
            tick_labels = ['A', 'B', 'C', 'D']
        cms_norm = cms.astype('float') / cms.sum(axis=1)[:, np.newaxis]
        avg_score = np.mean(accs)
        std_score = np.std(accs)
        combined = np.array([[f"{cms_val} ({cms_norm_val:.2f})" for cms_val, cms_norm_val in zip(cms_row, cms_norm_row)] for cms_row, cms_norm_row in zip(cms, cms_norm)])
        
        plt.figure()
        if XAI:
            sn.heatmap(cms, annot=combined, cbar=False, cmap='Blues', xticklabels=tick_labels, yticklabels=tick_labels, fmt='')
        else:
            sn.heatmap(cms, annot=combined, cbar=False, cmap='Reds', xticklabels=tick_labels, yticklabels=tick_labels, fmt='')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        if XAI:
            plt.title('RFE-SHAP (Pooled Confusion Matrix)')
            #plt.title('Cross-validated Accuracy: {:.3f} +- {:.3f}'.format(avg_score, std_score))
        else:
            plt.title('RFE (Pooled Confusion Matrix)')
            #plt.title('Cross-validated Accuracy: {:.3f} +- {:.3f}'.format(avg_score, std_score))

        if XAI:
            plt.savefig(export_dir + '/shap_norm_cm_{}.png'.format(SEED), dpi=400)
        else:
            plt.savefig(export_dir + '/norm_cm_{}.png'.format(SEED), dpi=400)