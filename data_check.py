import numpy as np
import matplotlib.pyplot as plt
import os

from skimage import exposure

'''
Script to visualize density mask over "processed" projection to check data
'''

if __name__ == '__main__':
    data_dir = "\\\\rad-maid-002/D/Users/vincent/prospr_data/data/all"
    export_dir = './data_check/'
    data = os.listdir(data_dir)
    for sample in data:
        sample_name = sample.split('.npz')[0]
        sample_npz = np.load(data_dir + '/' + sample)
        proj = sample_npz['proj_2D']
        proj_eq = (exposure.equalize_hist(proj)*255).astype(np.uint8)
        mask = sample_npz['arg_max']
        mask_masked = np.where(mask==0, np.nan, mask)
        
        fig, ax = plt.subplots(1, 2, tight_layout=True)
        ax[0].imshow(proj_eq, 'gray_r')
        ax[0].axis('off')

        ax[1].imshow(proj_eq, 'gray_r')
        ax[1].imshow(mask_masked, 'bwr', alpha=0.3)
        ax[1].axis('off')

        fig.suptitle(sample_name)

        plt.tight_layout()
        plt.savefig(export_dir + sample_name, dpi=300)
        plt.close()