import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, RocCurveDisplay

from auc_delong_xu import delong_roc_test

if __name__ == '__main__':
    rfe_probs_path = './results/cv_results/pooled_probs_42.csv'
    rfe_probs = pd.read_csv(rfe_probs_path)
    rfe_y = rfe_probs['label']
    rfe_probs = rfe_probs.drop(columns=['label'])

    rfe_shap_probs_path = './results/cv_results/shap_pooled_probs_42.csv'
    rfe_shap_probs = pd.read_csv(rfe_shap_probs_path)
    rfe_shap_y = rfe_shap_probs['label']
    rfe_shap_probs = rfe_shap_probs.drop(columns=['label'])

    if rfe_y.equals(rfe_shap_y):
        y = rfe_y
    else:
        print('ERROR: order of samples not the same')

    target_names = ['Density Grade A', 'Density Grade B', 'Density Grade C', 'Density Grade D']

    num_classes = 4
    y_onehot = label_binarize(y, classes=[0, 1, 2, 3])

    fig, ax = plt.subplots()

    for c in range(num_classes):
        p_value = 10 ** delong_roc_test(y_onehot[:, c], rfe_probs[str(c)], rfe_shap_probs[str(c)])
        print('Delong Test p-value for {} = {}'.format(target_names[c], p_value[0][0]))

    print(1)
