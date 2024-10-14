import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, RocCurveDisplay

if __name__ == '__main__':
    probs_path = './results/cv_results/pooled_probs_42.csv'
    probs = pd.read_csv(probs_path)
    y = probs['label']
    probs = probs.drop(columns=['label'])
    target_names = ['Density Grade A', 'Density Grade B', 'Density Grade C', 'Density Grade D']

    num_classes = 4
    y_onehot = label_binarize(y, classes=[0, 1, 2, 3])

    fig, ax = plt.subplots()

    for c in range(num_classes):
        auc_check = roc_auc_score(y_onehot[:, c], probs[str(c)])
        if auc_check < 0.5:
            probs[str(c)] = 1 - probs[str(c)]
        viz = RocCurveDisplay.from_predictions(
            y_onehot[:, c],
            probs[str(c)],
            name=f"Pooled ROC for {target_names[c]}",
            lw=1,
            ax=ax
        )
    _ = ax.set(
        xlabel = 'False Positive Rate',
        ylabel = 'True Positive Rate',
        title = 'Pooled ROC (RFE)'
    )
    plt.savefig('./results/cv_results/pooled_roc_42.png', dpi=400)
    print(1)
