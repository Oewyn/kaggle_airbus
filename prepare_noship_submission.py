import numpy as np
import pandas as pd
from keras import models, layers
from losses import *

MONTAGE_SIZE=[4, 4]
AUGMENT_IMG=True
CROP=True
FILTERS=8

expansion_schedule3 = [True, False, False, False,
                       True, False, False, False,
                       True, False, False, False,
                       True, False, False, False,
                       True, False, False, False]

VIRTUAL_BATCH_SIZE=8
GPU_BATCH_SIZE=16
SGDR_CYCLE_LENGTH=1
SGDR_MULT_FACTOR=2.
SGDR_LR_DECAY=1.0
TARGET_SIZE=768
VALID_IMG_COUNT=128
NUM_EPOCHS=7
MINIBATCH_SIZE=5000
SGDR_MIN_LEARNING_RATE=1e-6
SGDR_MAX_LEARNING_RATE=1e-4
PATIENCE=128
model_load=None
MODEL_NAME='SGDR_separable_8'
BETA=5.
KAPPA=1.
UNET_DEPTH=20
expansion_schedule=expansion_schedule3
FIRE=False
BAT_NORM_LAYERS=False                                                                             
EMPTY_SUBMISSION=False

import numpy as np
from keras import models, layers
from losses import *
from model import get_unet_sq
from optimizers import AdamAccumulate
from keras.optimizers import SGD

import os
import math

from tqdm import tqdm
from skimage.morphology import binary_opening, disk, label

from img_gen import make_test_gen, ThreadedWorker
from rle import multi_rle_encode, rle_encode

test_image_dir = '../data2/v2/test/'
img_list = os.listdir(test_image_dir)

out_pred_rows = []

if not EMPTY_SUBMISSION:
    import gc; gc.enable() # memory is tight

    seg_model = models.load_model('models/768_768_noship_resnet34_ohem_dice_best_loss_weights_seg_model.h5', custom_objects={'AdamAccumulate': AdamAccumulate()})
    seg_model.summary()

    seg_model.load_weights('checkpoints_noship/768_768_noship_resnet34_ohem_dice_best_loss_weights_best_acc_weights.h5')

    test_gen = ThreadedWorker(make_test_gen(img_list, test_image_dir, GPU_BATCH_SIZE, target_size=TARGET_SIZE))

    total = math.ceil(len(img_list)/GPU_BATCH_SIZE)

    print('total batches: {}'.format(total))

    full_submission = np.ones((768, 768, 1))

    for i in tqdm(range(total)):
        test_batch, test_names = next(test_gen)
        test_seg = seg_model.predict(test_batch, batch_size=GPU_BATCH_SIZE)
        #test_pred = np.greater(test_seg, 0.75)
        test_pred = test_seg

        for pred, c_img_name in zip(test_pred[0:], test_names):
            
            '''
            if pred:
                c_rle = rle_encode(full_submission.T)
                out_pred_rows += [{'ImageId': c_img_name, 'EncodedPixels': c_rle}]
            else:
                out_pred_rows += [{'ImageId': c_img_name, 'EncodedPixels': None}]
            '''
            out_pred_rows += [{'ImageId': c_img_name, 'Probability': pred[0]}]

        if i % 100 == 0:
            gc.collect()

else:
    out_pred_rows = [{'ImageId': c_img_name, 'EncodedPixels': None} for c_img_name in img_list]

#submission_df = pd.DataFrame(out_pred_rows)[['ImageId', 'EncodedPixels']]
submission_df = pd.DataFrame(out_pred_rows)[['ImageId', 'Probability']]
submission_df.to_csv('csvs/768_resnet_noship_ohem_dice_prob.csv', index=False)
print(submission_df.head())
# In[ ]:
