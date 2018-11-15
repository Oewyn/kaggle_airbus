import numpy as np
import pandas as pd

import os

from tqdm import tqdm
from skimage.morphology import binary_opening, disk, label

from rle import multi_rle_encode, rle_encode

test_image_dir = '../data2/v2/test/'
img_list = os.listdir(test_image_dir)

# invert ship/noship prediction to probe LB for how many FN detector has
INVERT_PRED=False

total = len(img_list)

out_pred_rows = []

predictions = pd.read_csv('csvs/768_resnet_noship_prob.csv')

THRESHOLD = 0.5

if not INVERT_PRED:
    has_ship_set = set(predictions.loc[predictions['Probability'] > THRESHOLD]['ImageId'].tolist())
else:
    has_ship_set = set(predictions.loc[predictions['Probability'] <= THRESHOLD]['ImageId'].tolist())

full_submission = rle_encode(np.ones((768, 768, 1)).T)

for i in tqdm(range(total)):
    if img_list[i] in has_ship_set:
        out_pred_rows += [{'ImageId': img_list[i], 'EncodedPixels': full_submission}]
    else:
        out_pred_rows += [{'ImageId': img_list[i], 'EncodedPixels': None}]


submission_df = pd.DataFrame(out_pred_rows)[['ImageId', 'EncodedPixels']]
submission_df.to_csv('csvs/05_thresh_768_resnet_noship.csv', index=False)
print(submission_df.head())
# In[ ]:
