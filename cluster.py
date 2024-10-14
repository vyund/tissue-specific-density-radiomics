import os

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sn

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import umap

from utils import *

# suppress warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
    warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
    warnings.simplefilter('ignore', category=FutureWarning)

'''
Script to perform UMAP/tSNE unsupervised clustering
'''

with open('exclude.txt') as f:
    exclude = f.read().splitlines()
EXCLUDE = [e.split(' - ')[0] for e in exclude]

GRADE_DICT = {
        'a': 0,
        'b': 1,
        'c': 2,
        'd': 3,
    }
GRADE_DICT = dict((v, k.upper()) for k, v in GRADE_DICT.items())

CLUSTER_ALG = 'umap'
FT_SET = 'discrete'

SELECTED_FT_SET = 'shap_b' #TODO: rfe, shap_b, shap_m (binary selection vs multiclass selection)
#TODO: binary = top 5 features for each class, then remove duplicates
#TODO: multiclass = features above 0.10 mean abs SHAP value

LABEL = 'density_grade'

AGG = True

AFTER = True
SAVE_FIG = True
ANNOTATE = False

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

    if LABEL == 'density_grade':
        if SUB_GRADE:
            colorbar.set_ticklabels([SUB_GRADE_DICT[i] for i in range(ncolors)])
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

def update_annot(ind):
    pos = sc.get_offsets()[ind['ind'][0]]
    annot.xy = pos

    if LABEL == 'density_grade':
        if SUB_GRADE:
            text = '{} : {}'.format(' '.join([SUB_GRADE_DICT[y[i]] for i in ind['ind']]), ' '.join([ids[i] for i in ind['ind']]))
        else:
            text = '{} : {}'.format(' '.join([GRADE_DICT[y[i]] for i in ind['ind']]), ' '.join([ids[i] for i in ind['ind']]))
        
    else:
        text = '{} : {}'.format(' '.join([str(y[i]) for i in ind['ind']]), ' '.join([ids[i] for i in ind['ind']]))

    annot.set_text(text)
    #annot.get_bbox_patch().set_facecolor(cmap(y[ind['ind'][0]]))

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

if __name__ == '__main__':
    data_path = './extracted_fts/extracted_fts_{}.csv'.format(FT_SET) #after feature selection
    #data_path = './extracted_fts_all.csv' #original image features

    label_path = './labels/reports.csv'
    #export_dir = './cluster_{}_{}_{}'.format(CLUSTER_ALG, FT_SET, SELECTED_FT_SET)
    export_dir = './results/cluster'

    if AFTER:
        #selected_fts_path = './selected_fts_{}_{}.csv'.format(FT_SET, SELECTED_FT_SET)
        selected_fts_path = './shap_cv/selected_fts_0.csv' #temp using 0
        selected_fts = pd.read_csv(selected_fts_path)['0'].to_list()

    if SAVE_FIG:
        if not os.path.exists(export_dir):
            os.mkdir(export_dir)

    data = pd.read_csv(data_path)
    data = data[~data['sample_name'].isin(EXCLUDE)]
    #data = data.drop('Unnamed: 0', axis=1) #TODO: not needed for new combined fts (discrete + combined) as of 8/5/23
    #labels = pd.read_excel(label_path)
    labels = pd.read_csv(label_path)

    #data_filtered = filter_single_view(data, view='MLO')
    labels_filtered = select_label(labels, label=LABEL)

    split_samples(data)

    if AGG:
        data_agg = aggregate_views(data, aggregate_on='sample_id', method='median')
        X = pd.merge(data_agg.astype({'sample_id': 'str'}), labels_filtered.astype({'DummyID': 'str'}).rename(columns={'DummyID': 'sample_id'}), on='sample_id')
    else:
        data = data.drop(['sample_name', 'view'], axis=1)
        X = pd.merge(data.astype({'sample_id': 'str'}), labels_filtered.astype({'DummyID': 'str'}).rename(columns={'DummyID': 'sample_id'}), on='sample_id')
    
    y = X[LABEL]
    ids = X['sample_id']

    X = X.drop(columns=['sample_id', LABEL])

    #X = X.loc[:, ~X.columns.str.contains('shape2D')] #remove shape features to check

    X, pruned_var = prune_var(X)
    X, pruned_corr = prune_corr(X)
    X[X.columns] = StandardScaler().fit_transform(X)

    X = X.loc[:, (X != 0).any(axis=0)] # remove columns with all 0

    if SELECTED_FT_SET == 'shap_b' and AFTER:
        selected_fts = [*set(selected_fts)]
        print(len(selected_fts))

    if AFTER:
        X = X[selected_fts]
    
    num_classes = y.value_counts().shape[0]

    cmap = mpl.colormaps.get_cmap('plasma')

    for num_neighbors in range(40, 64, 5):
        if num_neighbors == 0:
            num_neighbors = 1

        if CLUSTER_ALG == 'tsne':
            model = TSNE(n_components=2, perplexity=num_neighbors, learning_rate='auto', n_iter=10000, metric='euclidean', random_state=42, verbose=1)
        elif CLUSTER_ALG == 'umap':
            model = umap.UMAP(n_components=2, n_neighbors=num_neighbors, min_dist=0.1, random_state=42)

        embedding = model.fit_transform(X)

        fig, ax = plt.subplots()
        ax.set_facecolor('black')
        sc = plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap=cmap, s=25)

        if LABEL != 'case_scores':
            colorbar_index(ncolors=num_classes, cmap=cmap)
        else:
            plt.colorbar(sc)

        if ANNOTATE:
            annot = ax.annotate('', xy=(0,0), xytext=(10, 12), textcoords='offset points', color='black',
                                bbox=dict(boxstyle='round', fc='w'), arrowprops=dict(arrowstyle='->', color='w'))
            annot.set_visible(False)
            fig.canvas.mpl_connect('motion_notify_event', hover)

        #plt.suptitle('{} on {} - {}_{}'.format(CLUSTER_ALG.upper(), LABEL, FT_SET, SELECTED_FT_SET), fontsize=18)
        #plt.title('N-Neighbors = {}'.format(num_neighbors))
        if AFTER:
            plt.title('UMAP after Feature Selection (neighbors = {})'.format(num_neighbors), fontsize=14)
        else:
            plt.title('UMAP before Feature Selection (neighbors = {})'.format(num_neighbors), fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()

        if SAVE_FIG:
            #plt.savefig('{}/{}_{}_{}_{}'.format(export_dir, CLUSTER_ALG, FT_SET, SELECTED_FT_SET, num_neighbors), dpi=400) #TODO: replace label if analysis changes
            if AFTER:
                plt.savefig('{}/umap_post_{}.png'.format(export_dir, num_neighbors), dpi=400)
            else:
                plt.savefig('{}/umap_pre_{}.png'.format(export_dir, num_neighbors), dpi=400)
        else:
            plt.show()

        plt.close()

    print(1)