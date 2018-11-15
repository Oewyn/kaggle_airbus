import numpy as np
import pandas as pd

import os

from tqdm import tqdm
from skimage.morphology import binary_opening, disk, label

from rle import multi_rle_encode, rle_encode

test_image_dir = '../data2/v2/test/'
img_list = os.listdir(test_image_dir)

total = len(img_list)

out_pred_rows = []

stage1_predictions = pd.read_csv('csvs/768_resnet_noship_ohem_dice_prob.csv')
stage2_predictions = pd.read_csv('submissions/resnet34_768_ohem_best_loss_th_05_so_60_submission.csv')

THRESHOLD = 0.5

has_ship_set = set(stage1_predictions.loc[stage1_predictions['Probability'] > THRESHOLD]['ImageId'].tolist())

for i in tqdm(range(total)):
    if img_list[i] in has_ship_set:
        stage2_rows = stage2_predictions.loc[stage2_predictions['ImageId'] == img_list[i]]
        out_pred_rows += stage2_rows.filter(items=['ImageId', 'EncodedPixels']).to_dict('records')
    else:
        out_pred_rows += [{'ImageId': img_list[i], 'EncodedPixels': None}]


submission_df = pd.DataFrame(out_pred_rows)[['ImageId', 'EncodedPixels']]
submission_df.to_csv('submissions/2stage_resnet34_ohem_noship_050_ohem_dice_best_f2_th_05_so_60_TTA.csv', index=False)
print(submission_df.head())
# In[ ]:
