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
GPU_BATCH_SIZE=8
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
TTA=True
TTA_INTERSECT_RATIO=1.0
THRESHOLD=0.5
SMALL_OBJ_SIZE=60

import numpy as np
from keras import models, layers
from losses import *
from model import get_resnet34
from optimizers import AdamAccumulate
from keras.optimizers import SGD

import os
import math

from tqdm import tqdm
from skimage.morphology import binary_opening, disk, label

from img_preprocess import tta_batch, tta_batch_reverse
from img_gen import make_test_gen, ThreadedWorker
from rle import instance_segment_mask, rle_encode

test_image_dir = '../data2/v2/test/'
img_list = os.listdir(test_image_dir)

out_pred_rows = []

if not EMPTY_SUBMISSION:
    import gc; gc.enable() # memory is tight


    seg_model = get_resnet34(input_shape=(TARGET_SIZE, TARGET_SIZE, 3))
    seg_model.summary()

    seg_model.load_weights('checkpoints/768_768_resnet34_ohem_dice_best_loss_weights.h5')

    test_gen = ThreadedWorker(make_test_gen(img_list, test_image_dir, GPU_BATCH_SIZE))

    total = math.ceil(len(img_list)/GPU_BATCH_SIZE)

    print('total batches: {}'.format(total))

    for i in tqdm(range(total)):
        test_batch, test_names = next(test_gen)
        if not TTA:
            test_seg = seg_model.predict(test_batch, batch_size=GPU_BATCH_SIZE)
            test_pred = np.greater(test_seg, THRESHOLD).astype('float32')[0:,...]
        else:
            tta_x = tta_batch(test_batch)
            first_segs = [seg_model.predict(i, batch_size=GPU_BATCH_SIZE) for i in tta_x]

            aligned_segs = np.stack(tta_batch_reverse(first_segs), 3)
            avg_seg = np.sum(aligned_segs, axis=3, keepdims=True)

            num_augs = aligned_segs.shape[3]
            comparison = THRESHOLD * TTA_INTERSECT_RATIO * num_augs

            test_pred = np.greater(avg_seg, comparison).astype('float32')[0:,...]

        for pred, c_img_name in zip(test_pred[0:,:,:,:], test_names):
            
            # transpose rows/cols due to airbus data being col major order
            ships_pred = instance_segment_mask(pred, smallest_size=SMALL_OBJ_SIZE)
            if len(ships_pred) > 0:
                for ship in ships_pred:
                    c_rle = rle_encode(np.transpose(ship, axes=[1,0,2]))
                    out_pred_rows += [{'ImageId': c_img_name, 'EncodedPixels': c_rle}]
            else:
                out_pred_rows += [{'ImageId': c_img_name, 'EncodedPixels': None}]

        if i % 100 == 0:
            gc.collect()

else:
    out_pred_rows = [{'ImageId': c_img_name, 'EncodedPixels': None} for c_img_name in img_list]

submission_df = pd.DataFrame(out_pred_rows)[['ImageId', 'EncodedPixels']]
submission_df.to_csv('resnet34_768_ohem_best_loss_th_05_so_60_submission.csv', index=False)
print(submission_df.head())
# In[ ]:
