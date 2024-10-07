import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import SimpleITK as sitk
import radiomics
from radiomics import featureextractor

'''
Script to extract tissue-specific radiomic features using PyRadiomics
'''

with open('exclude.txt') as f:
    exclude = f.read().splitlines()
EXCLUDE = [e.split(' - ')[0] for e in exclude]

TISSUE_TYPE = 'adipose'

def get_tissue_mask(mask, tissue_type):
    # 0 = air, 127 = adipose, 254 = dense
    if tissue_type == 'dense':
        thresh = 254
    elif tissue_type == 'adipose':
        thresh = 127
    
    if tissue_type != 'all':
        mask[mask != thresh] = 0
        mask[mask == thresh] = 1
    else:
        mask[mask > 0] = 1
        # air should already be 0

    return mask

if __name__ == '__main__':
    data_dir = "\\\\rad-maid-002/D/Users/vincent/prospr_data/data/all"
    export_dir = './extracted_features/'

    params = './params/params.yaml'
    extractor = featureextractor.RadiomicsFeatureExtractor(params)

    extracted_fts = pd.DataFrame()

    #TODO: make extraction log text file (date of extraction, package versions, radiomic features, etc.)

    #75562677_L_CC... (extraction ended here --> merge adipose_temp_1400 with new)
    start_i = 1400
    data = os.listdir(data_dir)
    data = data[start_i:]
    for i, sample in enumerate(data, start=start_i):
        sample_name = sample.split('.npz')[0]

        if sample_name in EXCLUDE:
            print('skipping {}...'.format(sample_name))
            continue
        print('extracting {}...'.format(sample_name))

        sample_npz = np.load(data_dir + '/' + sample)

        img = sitk.GetImageFromArray(sample_npz['proj_2D']) #TODO: check what the correct array is
        mask = sitk.GetImageFromArray(get_tissue_mask(sample_npz['arg_max'], TISSUE_TYPE))

        extracted = extractor.execute(img, mask, voxelBased=False)

        info = {k:v for k,v in extracted.items() if 'diagnostic' in k}

        if i == 0:
            version_info = {k:v for k,v in info.items() if 'original' not in k}
            pd.Series(version_info).to_csv(export_dir + 'version_info_{}.csv'.format(TISSUE_TYPE))

        fts = {k:float(v) for k,v in extracted.items() if not 'diagnostic' in k} # remove diagnostic features
        sample_df = pd.DataFrame(fts, index=[0])
        sample_df.insert(0, 'sample_name', sample_name)

        extracted_fts = pd.concat([extracted_fts, sample_df], ignore_index=True)

        if i != start_i and i % 200 == 0:
            extracted_fts.to_csv(export_dir + 'extracted_fts_{}_temp_{}.csv'.format(TISSUE_TYPE, i))
    
    print('extraction for {} tissue complete...'.format(TISSUE_TYPE))
    extracted_fts.to_csv(export_dir + 'extracted_fts_{}.csv'.format(TISSUE_TYPE))
