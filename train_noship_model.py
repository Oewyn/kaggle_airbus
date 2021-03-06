from train_ship_noship import train

'''
train(VIRTUAL_BATCH_SIZE=8, GPU_BATCH_SIZE=3, SGDR_CYCLE_LENGTH=10, SGDR_MULT_FACTOR=2.,
      SGDR_LR_DECAY=1.0, TARGET_SIZE=768, VALID_IMG_COUNT=256, NUM_EPOCHS=70, MINIBATCH_SIZE=100,
      SGDR_MIN_LEARNING_RATE=1e-6, SGDR_MAX_LEARNING_RATE=1e-5, PATIENCE=128, model_load='checkpoints/768_768_resnet34_3_sr01_br75_3_best_f2_weights.h5',
      MODEL_NAME='noship_resnet34_3_sr01_br75_3_best_f2', BETA=8., KAPPA=1., UNET_DEPTH=20)

train(VIRTUAL_BATCH_SIZE=8, GPU_BATCH_SIZE=3, SGDR_CYCLE_LENGTH=20, SGDR_MULT_FACTOR=2.,
      SGDR_LR_DECAY=2.0, TARGET_SIZE=768, VALID_IMG_COUNT=512, NUM_EPOCHS=60, MINIBATCH_SIZE=100,
      SGDR_MIN_LEARNING_RATE=1e-7, SGDR_MAX_LEARNING_RATE=5e-6, PATIENCE=128, model_load='checkpoints/768_768_resnet34_3_sr01_br75_3_best_f2_weights.h5',
      MODEL_NAME='noship_resnet34_3_sr01_br75_3_best_f2_2', BETA=8., KAPPA=1., UNET_DEPTH=20)
'''

train(VIRTUAL_BATCH_SIZE=18, GPU_BATCH_SIZE=3, SGDR_CYCLE_LENGTH=10, SGDR_MULT_FACTOR=2.,
      SGDR_LR_DECAY=0.6, TARGET_SIZE=768, VALID_IMG_COUNT=256, NUM_EPOCHS=70, MINIBATCH_SIZE=400,
      SGDR_MIN_LEARNING_RATE=5e-7, SGDR_MAX_LEARNING_RATE=5e-6, PATIENCE=128, model_load='checkpoints/768_768_resnet34_ohem_dice_best_loss_weights.h5',
      MODEL_NAME='noship_resnet34_ohem_dice_best_loss_weights', BETA=8., KAPPA=1., UNET_DEPTH=20, OHEM=True)
