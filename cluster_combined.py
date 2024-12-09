import os

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import umap

from select_fts import load_data
from utils import *

# suppress warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
    warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
    warnings.simplefilter('ignore', category=FutureWarning)

'''
Script to perform UMAP/tSNE unsupervised clustering on combined data (i + ii)
'''

CLUSTER_ALG = 'umap'
FT_SET = 'discrete'
LABEL = 'density_grade'
XAI = True
DATASET = False # colormap by dataset instead of class
AFTER = True
SAVE_FIG = True

RANDOM = False
if RANDOM:
    SEED = np.random.randint(0, 1e5)
else:
    SEED = 42
 
GRADE_DICT = {
        'I - A': 0,
        'II - A': 4,
        'I - B': 1,
        'II - B': 5,
        'I - C': 2,
        'II - C': 6,
        'I - D': 3,
        'II - D': 7
    }

DATASET_DICT = {
    'I': 0,
    'II': 1
}

GRADE_DICT = dict((v, k.upper()) for k, v in GRADE_DICT.items())
DATASET_DICT = dict((v, k.upper()) for k, v in DATASET_DICT.items())

def filter_single_view(data, view='MLO'):
    return data[data['sample_name'].str.contains(view)]

def colorbar_index(ncolors, cmap):
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable, ax=plt.gca())
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.ax.zorder = -1
    colorbar.ax.invert_yaxis()

    if LABEL == 'density_grade':
        if DATASET:
            colorbar.set_ticklabels([DATASET_DICT[i] for i in range(ncolors)])
        else:
            colorbar.set_ticklabels([GRADE_DICT[i] for i in range(ncolors)])
    else:
        colorbar.set_ticklabels(str(i) for i in range(ncolors))

def cmap_discretize(cmap, N):
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki]) for i in range(N+1)]

    return mcolors.LinearSegmentedColormap(cmap.name + "%d"%N, cdict, 1024)

if __name__ == '__main__':
    data_path = './extracted_fts/extracted_fts_{}.csv'.format(FT_SET) #all features
    label_path = './labels/reports.csv'
    exclude_path = './exclude.txt'
    label_type = 'density_grade'

    data_path_val = './ext_val/extracted_fts_{}.csv'.format(FT_SET)
    label_path_val = './ext_val/reports.csv' 
    exclude_path_val = './ext_val/exclude.txt'
    label_type_val = 'Density_Overall'

    export_dir = './results/cluster_combined'
     
    X, y, _, _, _, _ = load_data(data_path, label_path, exclude_path, label_type, scale=False)
    X_val, y_val, _, _, _, _ = load_data(data_path_val, label_path_val, exclude_path_val, label_type_val, prune=False, scale=False)

    # match features if !=
    X_val = X_val[X_val.columns.intersection(X.columns)]

    scaler = StandardScaler()
    X[X.columns] = scaler.fit_transform(X)
    scaler_val = StandardScaler()
    X_val[X_val.columns] = scaler_val.fit_transform(X_val)

    X_combined = pd.concat([X, X_val], keys=['i', 'ii']).reset_index()
    X_combined = X_combined.drop(['level_0', 'level_1'], axis=1)


    y_paired = y.map({0: 0, 1: 1, 2: 2, 3: 3})
    y_val_paired = y_val.map({0: 4, 1: 5, 2: 6, 3: 7})

    y_combined = pd.concat([y_paired, y_val_paired], keys=['i', 'ii']).reset_index()
    
    y_dataset = y_combined['level_0'].map({'i': 0, 'ii': 1})
    y_combined = y_combined.drop(['level_0', 'level_1'], axis=1)

    if DATASET:
        num_classes = 2
    else:
        num_classes = 4
    
    if AFTER:
        if XAI:
            selected_fts_path = './selected_fts/i_shap_selected_fts_{}.csv'.format(SEED)
            selected_fts_path_val = './selected_fts/ii_shap_selected_fts_{}.csv'.format(SEED)
        else:
            selected_fts_path = './selected_fts/i_selected_fts_{}.csv'.format(SEED)
            selected_fts_path_val = './selected_fts/ii_selected_fts_{}.csv'.format(SEED)
        
        selected_fts = pd.read_csv(selected_fts_path)['0'].to_list()
        selected_fts = [*set(selected_fts)]
        X_combined = X_combined[selected_fts]

    if SAVE_FIG:
        if not os.path.exists(export_dir):
            os.mkdir(export_dir)

    if DATASET:
        cmap = mpl.colormaps.get_cmap('bwr')
    else:
        cmap1 = mpl.colormaps.get_cmap('Set2') #I
        cmap2 = mpl.colormaps.get_cmap('Pastel2') #II

        colors1 = cmap1.colors[0:num_classes]
        colors2 = cmap2.colors[0:num_classes]
        colors = np.vstack((colors1, colors2))

        cmap = mcolors.ListedColormap(colors, name='combined')

    for num_neighbors in range(55, 59, 5):
        if num_neighbors == 0:
            num_neighbors = 1

        if CLUSTER_ALG == 'tsne':
            model = TSNE(n_components=2, perplexity=num_neighbors, learning_rate='auto', n_iter=10000, metric='euclidean', random_state=SEED, verbose=1)
        elif CLUSTER_ALG == 'umap':
            model = umap.UMAP(n_components=2, n_neighbors=num_neighbors, min_dist=0.1, random_state=SEED)

        embedding = model.fit_transform(X_combined)

        fig, ax = plt.subplots()

        if DATASET:
            sc = plt.scatter(embedding[:, 0], embedding[:, 1], c=y_dataset, cmap=cmap, s=25)
        else:
            sc = plt.scatter(embedding[:, 0], embedding[:, 1], c=y_combined, cmap=cmap, s=25)

        if LABEL != 'case_scores':
            colorbar_index(ncolors=num_classes*2, cmap=cmap)
        else:
            plt.colorbar(sc)

        #plt.suptitle('{} on {} - {}_{}'.format(CLUSTER_ALG.upper(), LABEL, FT_SET, SELECTED_FT_SET), fontsize=18)
        #plt.title('N-Neighbors = {}'.format(num_neighbors))
        if AFTER:
            #plt.title('UMAP after Feature Selection (neighbors = {})'.format(num_neighbors), fontsize=14)
            plt.title('UMAP after Feature Selection', fontsize=14)
        else:
            #plt.title('UMAP before Feature Selection (neighbors = {})'.format(num_neighbors), fontsize=14)
            plt.title('UMAP before Feature Selection', fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()

        if SAVE_FIG:
            #plt.savefig('{}/{}_{}_{}_{}'.format(export_dir, CLUSTER_ALG, FT_SET, SELECTED_FT_SET, num_neighbors), dpi=400)
            if AFTER:
                plt.savefig('{}/umap_post_{}.png'.format(export_dir, num_neighbors), dpi=400)
            else:
                plt.savefig('{}/umap_pre_{}.png'.format(export_dir, num_neighbors), dpi=400)
        else:
            plt.show()
        plt.close()