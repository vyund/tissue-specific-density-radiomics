import os
import re

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sn

import ast

'''
Script to process results of nested cross-validation to generate confusion matrix and classification performance errorplot
'''

def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    return float(intersection) / union

if __name__ == '__main__':
    experiment_name = 'iwbi_icad_rad_results'
    #results_path = './results/nested_cv_results/nested_cv_results.csv'
    results_path = './results/{}/nested_cv_results_rad_no_d.csv'.format(experiment_name)
    results = pd.read_csv(results_path)

    num_classes = 3
    num_folds = 5

    #acc
    accs = results['acc'].to_list()
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)

    #auc
    aucs = [ast.literal_eval(i) for i in results['auc'].to_list()]

    class_mean_aucs = []
    class_std_aucs = []
    for c in range(num_classes):
        c_aucs = [auc[c] for auc in aucs]
        class_mean_aucs.append(np.mean(c_aucs))
        class_std_aucs.append(np.std(c_aucs))

    #cm
    
    cms = results['cm']
    if num_classes == 3:
        tick_labels = ['A', 'B', 'C']
        pooled_cm = [[0, 0, 0] for i in range(num_classes)]
    else:
        tick_labels = ['A', 'B', 'C', 'D']
        pooled_cm = [[0, 0, 0, 0] for i in range(num_classes)]
    for f in range(num_folds):
        stripped = '\n'.join(re.sub(r'\s+', ' ', line.replace('[ ', '[')) for line in cms[f].splitlines())
        c_cm = ast.literal_eval(stripped.replace(' ', ','))
        for i, (pcm, ccm) in enumerate(zip(pooled_cm, c_cm)):
            pooled_cm[i] = [x + y for x, y in zip(pcm, ccm)]

    
    plt.figure()
    sn.heatmap(pooled_cm, annot=True, cbar=False, cmap='Blues', xticklabels=tick_labels, yticklabels=tick_labels, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Pooled Confusion Matrix (Nested Cross-Validation)')
    plt.savefig('./results/{}/nested_cv_pooled_cm.png'.format(experiment_name), dpi=400)
    plt.close()

    print(1)
    
    '''
    #ft stability
    fts = [ast.literal_eval(i) for i in results['fts']]
    counts = {}
    for fold_fts in fts:
        for ft in fold_fts:
            ft_general = ft.split('_')
            del ft_general[1]
            f = '_'.join(ft_general)
            if ft not in counts:
                counts[ft] = 1
            else:
                counts[ft] += 1

    counts_sorted_keys = sorted(counts, key=counts.get, reverse=True)

    cols = ['tissue', 'filter', 'class', 'ft', 'count']
    ft_parts = []
    for key in counts_sorted_keys:
        parts = key.split('_')
        parts.append(counts[key])
        ft_parts.append(parts)
    
    fts_df = pd.DataFrame(ft_parts, columns=cols)
    fts_df_sorted = fts_df.sort_values(by=['count', 'ft'], ascending=False)
    print(1)
    '''
    # plot acc + auc errorplot
    plt.figure()
    if num_classes == 3:
        X = ['Accuracy', 'AUC - A', 'AUC - B', 'AUC - C']
    else:
        X = ['Accuracy', 'AUC - A', 'AUC - B', 'AUC - C', 'AUC - D']
    y = class_mean_aucs
    y.insert(0, mean_acc)
    y_err = class_std_aucs
    y_err.insert(0, std_acc)

    ax = plt.subplot(111)
    plt.scatter(X, y)
    plt.errorbar(X, y, yerr=y_err, fmt='o')
    plt.ylim(0.5, 1.05)
    plt.title('Classifier Performance (Nested Cross-Validation)')

    offset = 0.02
    for i in range(len(X)):
        plt.text(X[i], y[i] + offset, '  {:.3f}'.format(y[i]))

    ax.spines[['right', 'top']].set_visible(False)
    plt.savefig('./results/{}/nested_cv_clf_performance.png'.format(experiment_name), dpi=400)

    print(1)