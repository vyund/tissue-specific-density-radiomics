import pandas as pd

'''
Script to aggregate dense tissue features and adipose tissue features into "discrete" set
'''

if __name__ == '__main__':
    dense_ft_path = './extracted_features/extracted_fts_dense.csv'
    adipose_ft_path = './extracted_features/extracted_fts_adipose.csv'

    dense = pd.read_csv(dense_ft_path).drop('Unnamed: 0', axis=1).set_index('sample_name').add_prefix('dense_')
    adipose = pd.read_csv(adipose_ft_path).drop('Unnamed: 0', axis=1).set_index('sample_name').add_prefix('adipose_')

    discrete_tissue = dense.join(adipose).reset_index()

    discrete_tissue.to_csv('./extracted_features/extracted_fts_discrete.csv')

    print(1)