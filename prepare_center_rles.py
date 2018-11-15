from rle import *

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imsave
from tqdm import tqdm

np.random.seed(1)

ship_dir = '../data2'
train_image_dir = os.path.join(ship_dir, 'train')

masks = pd.read_csv(os.path.join('../data2/',
                                     'train_ship_segmentations.csv'))
print(masks.shape[0], 'masks found')
print(masks['ImageId'].value_counts().shape[0])

masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
# some files are too small/corrupt
unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(lambda c_img_id:
                                                               os.stat(os.path.join(train_image_dir,
                                                               c_img_id)).st_size/1024)
unique_img_ids = unique_img_ids[unique_img_ids['file_size_kb'] > 50] # keep only +50kb files
masks.drop(['ships'], axis=1, inplace=True)

out_center_masks = []

total = len(masks)

for idx, mask in tqdm(masks.iterrows(), total=total):
    c_rle = mask['EncodedPixels']
    if isinstance(c_rle, str):
        c_rle = rle_find_center(c_rle)
    else:
        c_rle = None
    out_center_masks += [{'ImageId': mask['ImageId'], 'EncodedPixels': c_rle}]

results_df = pd.DataFrame(out_center_masks)[['ImageId','EncodedPixels']]
results_df.to_csv('center_masks.csv', index=False)
print(results_df.head())
