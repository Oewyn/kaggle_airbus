
# coding: utf-8
import os
import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops
from pathlib import Path
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard

from airbus_train_test import AirbusTrainValGen
from model import get_resnet34_backbone
from losses import *
from optimizers import AdamAccumulate
from callbacks import *

def train(BAT_NORM_LAYERS=False, AUGMENT_IMG=True, CROP=True, MODEL_NAME='SGDR_0',
          VIRTUAL_BATCH_SIZE = 10, GPU_BATCH_SIZE = 2, FILTERS = 8, UNET_DEPTH = 6,
          SGDR_MIN_LEARNING_RATE = 1e-6, SGDR_MAX_LEARNING_RATE = 1e-5, SGDR_CYCLE_LENGTH=5,
          SGDR_LR_DECAY=0.6, SGDR_MULT_FACTOR=1.2, TARGET_SIZE = 768, VALID_IMG_COUNT = 64,
          NUM_EPOCHS = 95, MINIBATCH_SIZE = 100, PATIENCE=20, BETA=10., KAPPA=1.,
          model_load = None, expansion_schedule=None, FIRE=True, USE_LOSS_WEIGHTS=False,
          SHIP_GRAD_RATIO=0.5, BORDER_GRAD_RATIO=0.75, OHEM=False):

    print('====================================================================================================')
    print('Starting model: {}_{}_{}'.format(TARGET_SIZE, TARGET_SIZE, MODEL_NAME))
    print('====================================================================================================')

    np.random.seed(1)
    ship_dir = '../data2/v2/'
    train_image_dir = os.path.join(ship_dir, 'train')
    test_image_dir = os.path.join(ship_dir, 'test')

    # # Generate Validation Data
    train_val_gen = AirbusTrainValGen(train_image_dir,
                                      batch_size=GPU_BATCH_SIZE,
                                      valid_batch_size=VALID_IMG_COUNT,
                                      target_size=TARGET_SIZE,
                                      augment=False,
                                      crop=CROP,
                                      use_loss_weights=USE_LOSS_WEIGHTS,
                                      ship_grad_ratio=SHIP_GRAD_RATIO,
                                      border_grad_ratio=BORDER_GRAD_RATIO,
                                      online_hard_example_mining=OHEM,
                                      noship_detector=True)
    if OHEM:
        train_ids = train_val_gen.train_ids
    valid_x, valid_y = next(train_val_gen.valid_gen)

    gpu_batches_to_accum = int(VIRTUAL_BATCH_SIZE / GPU_BATCH_SIZE)
    optimizer = AdamAccumulate(lr=SGDR_MAX_LEARNING_RATE, accum_iters=gpu_batches_to_accum)
    from keras.losses import binary_crossentropy
    loss_fn = binary_crossentropy
    loss_fn_np = bce_np(scalar_loss=False)
    seg_model = get_resnet34_backbone(input_shape=(TARGET_SIZE, TARGET_SIZE, 3), loss=loss_fn, optimizer=optimizer, use_loss_weights=USE_LOSS_WEIGHTS)
    seg_model.summary()

    # load model for fine-tuning
    if model_load != None:
        seg_model.load_weights(model_load, by_name=True)
    seg_model.save('models/{}_{}_{}_seg_model.h5'.format(TARGET_SIZE, TARGET_SIZE, MODEL_NAME))

    best_acc_weight_path="checkpoints_noship/{}_{}_{}_best_acc_weights.h5".format(TARGET_SIZE, TARGET_SIZE, MODEL_NAME)
    last_weight_path="checkpoints_noship/{}_{}_{}_last_weights.h5".format(TARGET_SIZE, TARGET_SIZE, MODEL_NAME)

    best_acc_checkpoint = ModelCheckpoint(best_acc_weight_path, monitor='val_binary_accuracy', verbose=1, 
                                      save_best_only=True, mode='max', save_weights_only = True)

    last_checkpoint = ModelCheckpoint(last_weight_path, monitor='val_loss', verbose=0, 
                                      save_best_only=False, mode='min', save_weights_only = True)

    sgdrScheduler = SGDRScheduler(min_lr=SGDR_MIN_LEARNING_RATE, max_lr=SGDR_MAX_LEARNING_RATE, steps_per_epoch=MINIBATCH_SIZE, lr_decay=SGDR_LR_DECAY, cycle_length=SGDR_CYCLE_LENGTH, mult_factor=SGDR_MULT_FACTOR)

    early = EarlyStopping(monitor="val_loss", 
                          mode="min", 
                          patience=PATIENCE)

    tboard = LRTensorBoard(log_dir='./Graph/{}_{}_{}'.format(TARGET_SIZE, TARGET_SIZE, MODEL_NAME), histogram_freq=0,
            write_graph=False, write_images=False, log_every=10)

    callbacks_list = [best_acc_checkpoint, last_checkpoint, early, sgdrScheduler, tboard]

    if OHEM:
        callbacks_list.insert(0, OHEMEvaluator(train_ids, train_val_gen.ohem_input_queue, loss_fn_np))
        fetches = [tf.assign(callbacks_list[0].var_y_true, seg_model.targets[0], validate_shape=False, name='OHEM_y_true_assign'),
                   tf.assign(callbacks_list[0].var_y_pred, seg_model.outputs[0], validate_shape=False, name='OHEM_y_pred_assign')]

        seg_model._function_kwargs = {'fetches': fetches}

    seg_model.fit_generator(train_val_gen.train_gen,
                            verbose=1,
                            steps_per_epoch=MINIBATCH_SIZE, 
                            epochs=NUM_EPOCHS, 
                            validation_data=(valid_x, valid_y),
                            callbacks=callbacks_list,
                            workers=1)

    seg_model.save_weights('{}_{}_{}_last_checkpoint.h5'.format(TARGET_SIZE, TARGET_SIZE, MODEL_NAME))
    print('====================================================================================================')
    print('Finishing model: {}_{}_{}'.format(TARGET_SIZE, TARGET_SIZE, MODEL_NAME))
    print('====================================================================================================')
