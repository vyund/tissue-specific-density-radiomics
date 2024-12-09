import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

'''
Common functions
'''

FT_CLASSES = {'shape2D': [], 'firstorder': [], 'glcm': [], 'glrlm': [], 'glszm': [], 'gldm': []}
LABEL = 'density_grade'

def select_label(labels, label=LABEL):
    return labels[['DummyID', label]]

def split_samples(data):
    split = data['sample_name'].str.split('_', n=1, expand=True)
    data['sample_id'] = split[0]
    data['view'] = split[1]

    return data

def aggregate_views(data, aggregate_on='sample_id', method='mean'):
    if method == 'mean':
        return data.groupby(aggregate_on).mean().reset_index()
    elif method == 'median':
        return data.groupby(aggregate_on).median().reset_index()
    
def prune_var(data, thresh=0):
    variances = data.var()
    to_prune = variances[variances <= thresh].index
    X = data.drop(to_prune, axis=1)

    return X, to_prune

def prune_corr(data, thresh=0.95, plot=False):
    corr_matrix = data.corr()
    corr_matrix_abs = data.corr().abs()
    upper_tri = corr_matrix_abs.where(np.triu(np.ones(corr_matrix_abs.shape), k=1).astype(bool))

    to_prune = [col for col in upper_tri.columns if any(upper_tri[col] > thresh)]

    X = data.drop(to_prune, axis=1)

    if plot:
        fig, ax = plt.subplots(figsize=(20, 20))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        mat = ax.matshow(corr_matrix, cmap='coolwarm')
        fig.colorbar(mat, cax=cax, orientation='vertical')

        cols = corr_matrix.columns.to_list()
        ft_labels = [label.replace('original_', '') for label in cols]

        for ft_label in ft_labels:
            ft_class = ft_label.split('_')[0]
            if ft_class in FT_CLASSES:
                FT_CLASSES[ft_class].append(ft_label)

        ax.set_xticks(np.arange(0, corr_matrix.shape[0]), labels=ft_labels, rotation='vertical')
        ax.set_yticks(np.arange(0, corr_matrix.shape[1]), labels=ft_labels, rotation=0)

        plt.rcParams['xtick.labelsize'] = 'small'
        plt.rcParams['ytick.labelsize'] = 'small'
        plt.tight_layout()

        #plt.savefig('corr_matrix.png')
        plt.show()

    return X, to_prune