from losses import f2
from rle import *

import numpy as np 
import pandas as pd 
import keras.backend as K
from keras.callbacks import Callback, TensorBoard

""" F2 metric implementation for Keras models. Inspired from this Medium
article: https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2
Before we start, you might ask: this is a classic metric, isn't it already 
implemented in Keras? 
The answer is: it used to be. It has been removed since. Why?
Well, since metrics are computed per batch, this metric was confusing 
(should be computed globally over all the samples rather than over a mini-batch).
For more details, check this: https://github.com/keras-team/keras/issues/5794.
In this short code example, the F2 metric will only be called at the end of 
each epoch making it more useful (and correct).
"""

# Notice that since this competition has an unbalanced positive class
# (fewer ), a beta of 2 is used (thus the F2 score). This favors recall
# (i.e. capacity of the network to find positive classes). 

# Some default constants

START = 0.5
END = 0.95
STEP = 0.05
N_STEPS = int((END - START) / STEP) + 2
DEFAULT_THRESHOLDS = np.linspace(START, END, N_STEPS)
DEFAULT_LOGS = {}
FBETA_METRIC_NAME = "val_f2"

# Notice that this callback only works with Keras 2.0.0


class FBetaMetricCallback(Callback):

    def __init__(self, thresholds=DEFAULT_THRESHOLDS, batch_size=32, seg_threshold=0.5):
        self.thresholds = thresholds
        self.batch_size = batch_size
        self.seg_threshold = seg_threshold
        # Will be initialized when the training starts
        self.val_f2 = None

    def on_train_begin(self, logs=DEFAULT_LOGS):
        """ This is where the validation Fbeta
        validation scores will be saved during training: one value per
        epoch.
        """
        self.val_f2 = []

    # TODO optimize rle encode/decode loop and use labels directly
    def _score_per_img(self, y_true, y_pred):
        rles_true = multi_rle_encode(y_true)
        rles_pred = multi_rle_encode(y_pred)
        ships_true = [rle_decode(k) for k in rles_true]
        ships_pred = [rle_decode(k) for k in rles_pred]
        f2_score = f2(ships_true, ships_pred)

        '''
        intersection = np.sum(y_true.flatten() * y_pred.flatten())
        union = np.sum(y_true.flatten()) + np.sum(y_pred.flatten())
        dice = 2 * intersection / (union)

        print('f2 score: {} dice: {} num ships y_true: {} y_pred: {}'.format(f2_score, dice, len(rles_true), len(rles_pred)))
        '''
        return f2_score

    def on_epoch_end(self, epoch, logs=DEFAULT_LOGS):
        # save off _function_kwargs then restore due to error with OHEM callbacks
        saved_f_kwargs = self.model._function_kwargs
        self.model._function_kwargs = {}
        val_pred = np.greater(self.model.predict(self.validation_data[0], batch_size=self.batch_size), self.seg_threshold).astype(np.float32)
        self.model._function_kwargs = saved_f_kwargs

        val_true = self.validation_data[1]
        f2_score = np.mean([self._score_per_img(y_true, y_pred) for y_true, y_pred in zip(val_true[0:,...], val_pred[0:,...])])
        self.val_f2.append(f2_score)
        logs[FBETA_METRIC_NAME] = f2_score
        print("Current F2 metric is: {0:.3f}".format(f2_score))
        return

    def on_train_end(self, logs=DEFAULT_LOGS):
        """ Assign the validation Fbeta computed metric to the History object.
        """
        self.model.history.history[FBETA_METRIC_NAME] = self.val_f2

"""
Here is how to use this metric: 
Create a model and add the FBetaMetricCallback callback (with beta set to 2).
f2_metric_callback = FBetaMetricCallback(beta=2)
callbacks = [f2_metric_callback]
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                    nb_epoch=10, batch_size=64, callbacks=callbacks)
print(history.history.val_f2)
"""

import matplotlib.pyplot as plt

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


class LRFinder(Callback):
    
    '''
    A simple callback for finding the optimal learning rate range for your model + dataset. 
    
    # Usage
        ```python
            lr_finder = LRFinder(min_lr=1e-5, 
                                 max_lr=1e-2, 
                                 steps_per_epoch=np.ceil(epoch_size/batch_size), 
                                 epochs=3)
            model.fit(X_train, Y_train, callbacks=[lr_finder])
            
            lr_finder.plot_loss()
        ```
    
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 
        epochs: Number of epochs to run experiment. Usually between 2 and 4 epochs is sufficient. 
        
    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: https://arxiv.org/abs/1506.01186
    '''
    
    def __init__(self, min_lr=1e-5, max_lr=1e-2, steps_per_epoch=None, epochs=None):
        super().__init__()
        
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iterations = steps_per_epoch * epochs
        self.iteration = 0
        self.lr = min_lr
        self.lr_mult = (max_lr / min_lr) ** (1/self.total_iterations)
        self.history = {}
        
    def on_train_begin(self, logs=None):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.min_lr)
        
    def on_batch_end(self, epoch, logs=None):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.iteration += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.iteration)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
            
        self.lr *= self.lr_mult
        #print('new_lr = {}'.format(self.lr))
        K.set_value(self.model.optimizer.lr, self.lr)
 
    def plot_lr(self):
        '''Helper function to quickly inspect the learning rate schedule.'''
        plt.plot(self.history['iterations'], self.history['lr'])
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')
        
    def plot_loss(self):
        '''Helper function to quickly observe the learning rate experiment results.'''
        loss_avg = movingaverage(self.history['loss'], 10)
        #plt.plot(self.history['lr'], self.history['loss'])
        plt.plot(self.history['lr'], loss_avg)
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')


class SGDRScheduler(Callback):
    '''Cosine annealing learning rate scheduler with periodic restarts.
    # Usage
        ```python
            schedule = SGDRScheduler(min_lr=1e-5,
                                     max_lr=1e-2,
                                     steps_per_epoch=np.ceil(epoch_size/batch_size),
                                     lr_decay=0.9,
                                     cycle_length=5,
                                     mult_factor=1.5)
            model.fit(X_train, Y_train, epochs=100, callbacks=[schedule])
        ```
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 
        lr_decay: Reduce the max_lr after the completion of each cycle.
                  Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
        cycle_length: Initial number of epochs in a cycle.
        mult_factor: Scale epochs_to_restart after each full cycle completion.
    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: http://arxiv.org/abs/1608.03983
    '''
    def __init__(self,
                 min_lr,
                 max_lr,
                 steps_per_epoch,
                 lr_decay=1,
                 cycle_length=10,
                 mult_factor=2):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.steps_per_epoch = steps_per_epoch

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor

        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs={}):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        K.set_value(self.model.optimizer.lr, self.clr())

    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs={}):
        '''Set weights to the values from the end of the most recent cycle for best performance.'''
        self.model.set_weights(self.best_weights)

import tensorflow as tf

class LRTensorBoard(TensorBoard):
    def __init__(self, log_every=1, **kwargs):
        super().__init__(**kwargs)
        self.log_every = log_every
        self.counter = 0

    def on_batch_end(self, batch, logs=None):
        self.counter += 1
        if self.counter % self.log_every == 0:
            logs.update({'lr': K.eval(self.model.optimizer.lr)})
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.writer.add_summary(summary, self.counter)
            self.writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        for name, value in logs.items():
            if name in ['acc', 'loss', 'batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, self.counter)
        self.writer.flush()
        #super().on_epoch_end(epoch, logs)

import heapq

class OHEMEvaluator(Callback):
    def __init__(self, train_id_queue, out_candidates, loss_fn, num_examples=100, num_batches=50):
        super(OHEMEvaluator, self).__init__()
        self.var_y_true = tf.Variable(0., validate_shape=False, name='ohem_var_y_true')
        self.var_y_pred = tf.Variable(0., validate_shape=False, name='ohem_var_y_pred')
        self.train_id_queue = train_id_queue
        self.out_candidates = out_candidates
        self.candidates = []
        self.loss_fn = loss_fn
        self.num_examples = num_examples
        self.num_batches = num_batches
        self.batches_remaining = self.num_batches

    def on_batch_end(self, batch, logs=None):
        y_true = K.eval(self.var_y_true)
        y_pred = K.eval(self.var_y_pred)
        loss = self.loss_fn(y_true, y_pred)
        for idx, l in enumerate(loss.tolist()):
            winner = False
            if len(self.candidates) < self.num_examples:
                heapq.heappush(self.candidates, (l, self.train_id_queue[idx]))
            else:
                heapq.heappushpop(self.candidates, (l, self.train_id_queue[idx]))
                
        # remove this batch from train_id_queue
        del self.train_id_queue[:loss.shape[0]]

        self.batches_remaining -= 1
        if self.batches_remaining is 0:
            self.out_candidates.extend([ train_id for loss, train_id in self.candidates ])
            #print('out candidates: {}'.format(self.candidates))
            self.candidates.clear()
            self.batches_remaining = self.num_batches
