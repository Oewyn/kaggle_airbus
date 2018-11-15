import keras.backend as K
import numpy as np

def sigmoid_np(val):
    return 1/(1+np.exp(-val))

def iou_np(y_true, y_pred, eps=1e-6):
    y_t_flat = y_true.flatten()
    y_p_flat = y_pred.flatten()
    intersection = np.sum(y_t_flat * y_p_flat)
    union = np.sum(y_t_flat) + np.sum(y_p_flat) - intersection
    return np.mean( (intersection + eps) / (union + eps))

def iou(y_true, y_pred, eps=1e-6):
    y_t_flat = K.flatten(y_true)
    y_p_flat = K.flatten(y_pred)
    intersection = K.sum(y_t_flat * y_p_flat)
    union = K.sum(y_t_flat) + K.sum(y_p_flat) - intersection
    return K.mean( (intersection + eps) / (union + eps))

def iou_loss(y_true, y_pred):
    return 1. - iou(y_true, y_pred)

class bce():
    def __init__(self, use_loss_weights=False):
        self.__name__ = 'bce'
        self.use_loss_weights = use_loss_weights

    def __call__(self, y_true, y_pred):
        if not self.use_loss_weights:
            return K.mean(K.binary_crossentropy(y_true, y_pred))
        else:
            y_loss_weights = K.expand_dims(y_true[...,1], axis=-1)
            y_true = K.expand_dims(y_true[...,0], axis=-1)
            return K.mean(y_loss_weights * K.binary_crossentropy(y_true, y_pred))

class bce_np():
    def __init__(self, use_loss_weights=False, scalar_loss=True):
        self.__name__ = 'bce'
        self.use_loss_weights = use_loss_weights
        self.scalar_loss=scalar_loss

    def __call__(self, y_true, y_pred):
        if self.use_loss_weights:
            y_loss_weights = np.expand_dims(y_true[...,1], axis=-1)
            y_true = np.expand_dims(y_true[...,0], axis=-1)
            
        # taken from numpy_backend.py in Keras/backend
        y_pred_clip = np.clip(y_pred, 1e-7, 1-1e-7)
        y_pred_clip = np.log(y_pred_clip / (1 - y_pred_clip))
        
        loss = y_true * -np.log(sigmoid_np(y_pred_clip)) + (1 - y_true) * -np.log(1 - sigmoid_np(y_pred_clip))
        if self.use_loss_weights:
            loss = loss * y_loss_weights

        if self.scalar_loss:
            return np.mean(loss)
        else:
            return np.mean(loss, axis=tuple(range(1,loss.ndim)))

class dice_coef():
    def __init__(self, smooth=1, use_loss_weights=False, ignore_loss=False):
        self.smooth = smooth
        self.__name__ = 'sdice'
        self.use_loss_weights = use_loss_weights
        self.ignore_loss = ignore_loss

    def __call__(self, y_true, y_pred):
        if not self.use_loss_weights:
            y_t_flat = K.flatten(y_true)
            y_p_flat = K.flatten(y_pred)
            intersection = K.sum(y_t_flat * y_p_flat)
            union = K.sum(y_t_flat) + K.sum(y_p_flat)
            return (2. * intersection + self.smooth) / (union + self.smooth)
        elif not self.ignore_loss:
            y_loss_weights = K.expand_dims(y_true[...,1], axis=-1)
            y_true = K.expand_dims(y_true[...,0], axis=-1)
            y_t_flat = K.flatten(y_true)
            y_p_flat = K.flatten(y_pred)
            y_lw_flat = K.flatten(y_loss_weights)
            intersection = K.sum(y_t_flat * y_p_flat * y_lw_flat)
            union = K.sum(y_t_flat * y_lw_flat) + K.sum(y_p_flat * y_lw_flat)
            return (2. * intersection + self.smooth) / (union + self.smooth)
        else:
            y_loss_weights = K.expand_dims(y_true[...,1], axis=-1)
            y_true = K.expand_dims(y_true[...,0], axis=-1)
            y_t_flat = K.flatten(y_true)
            y_p_flat = K.flatten(y_pred)
            intersection = K.sum(y_t_flat * y_p_flat)
            union = K.sum(y_t_flat) + K.sum(y_p_flat)
            return (2. * intersection + self.smooth) / (union + self.smooth)

class dice_coef_np():
    def __init__(self, smooth=1, use_loss_weights=False, ignore_loss=False, scalar_loss=True):
        self.smooth = smooth
        self.__name__ = 'sdice'
        self.use_loss_weights = use_loss_weights
        self.ignore_loss = ignore_loss
        self.scalar_loss = scalar_loss

    def __call__(self, y_true, y_pred):
        if not self.use_loss_weights:
            intersection = np.sum(y_true * y_pred)
            union = np.sum(y_true) + np.sum(y_pred)
            loss = (2. * intersection + self.smooth) / (union + self.smooth)
        elif not self.ignore_loss:
            y_loss_weights = np.expand_dims(y_true[...,1], axis=-1)
            y_true = np.expand_dims(y_true[...,0], axis=-1)
            intersection = np.sum(y_true * y_pred * y_loss_weights)
            union = np.sum(y_true * y_loss_weights) + np.sum(y_pred * y_loss_weights)
            loss = (2. * intersection + self.smooth) / (union + self.smooth)
        else:
            y_true = np.expand_dims(y_true[...,0], axis=-1)
            intersection = np.sum(y_true * y_pred)
            union = np.sum(y_true) + np.sum(y_pred)
            loss = (2. * intersection + self.smooth) / (union + self.smooth)

        if self.scalar_loss:
            return np.mean(loss)
        else:
            return np.mean(loss, axis=tuple(range(1,loss.ndim)))

class dice_coef_loss():
    def __init__(self, smooth=1, use_loss_weights=False, ignore_loss=True):
        self.dice_coef=dice_coef(smooth=smooth, use_loss_weights=use_loss_weights, ignore_loss=True)
        self.__name__ = 'sdice_l'

    def __call__(self, y_true, y_pred):
        return 1 - self.dice_coef(y_true, y_pred)

class dice_coef_loss_np():
    def __init__(self, smooth=1, use_loss_weights=False, ignore_loss=True, scalar_loss=False):
        self.dice_coef=dice_coef_np(smooth=smooth, use_loss_weights=use_loss_weights, ignore_loss=True)
        self.__name__ = 'sdice_l'

    def __call__(self, y_true, y_pred):
        return 1 - self.dice_coef(y_true, y_pred)

import tensorflow as tf

import sys

class focal_loss():
    def __init__(self, gamma=2., alpha=.25, use_loss_weights=False, ignore_loss=True):
        self.gamma = gamma
        self.alpha = alpha
        self.__name__ = 'foc2_l'
        self.use_loss_weights=use_loss_weights
        self.ignore_loss=ignore_loss

    def __call__(self, y_true, y_pred):
        if self.use_loss_weights:
            y_loss_weights = K.expand_dims(y_true[...,1], axis=-1)
            y_true = K.expand_dims(y_true[...,0], axis=-1)
        '''
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(self.alpha * K.pow(1. - pt_1, self.gamma) * K.log(pt_1))-K.sum((1-self.alpha) * K.pow( pt_0, self.gamma) * K.log(1. - pt_0))
        '''
        max_val = K.clip(-y_pred, min_value = 0, max_value = sys.float_info.max)
        loss = y_pred - y_pred * y_true + max_val + K.log(K.exp(-max_val) + K.exp(-y_pred - max_val))

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = K.log(K.sigmoid(-y_pred * (y_true * 2 - 1)))
        loss = K.exp(invprobs * self.gamma) * loss
        
        if not self.use_loss_weights:
            return K.mean(loss)
        elif not self.ignore_loss:
            return K.mean(loss * y_loss_weights)
        else:
            return K.mean(loss)

class focal_loss_np():
    def __init__(self, gamma=2., alpha=.25, use_loss_weights=False, ignore_loss=True, scalar_loss=True):
        self.gamma = gamma
        self.alpha = alpha
        self.__name__ = 'foc2_l'
        self.use_loss_weights=use_loss_weights
        self.ignore_loss=ignore_loss
        self.scalar_loss=scalar_loss

    def __call__(self, y_true, y_pred):
        if self.use_loss_weights:
            y_loss_weights = np.expand_dims(y_true[...,1], axis=-1)
            y_true = np.expand_dims(y_true[...,0], axis=-1)
        '''
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(self.alpha * K.pow(1. - pt_1, self.gamma) * K.log(pt_1))-K.sum((1-self.alpha) * K.pow( pt_0, self.gamma) * K.log(1. - pt_0))
        '''
        max_val = np.clip(-y_pred, a_min = 0, a_max = sys.float_info.max)
        loss = y_pred - y_pred * y_true + max_val + np.log(np.exp(-max_val) + np.exp(-y_pred - max_val))

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = np.log(sigmoid_np(-y_pred * (y_true * 2 - 1)))
        loss = np.exp(invprobs * self.gamma) * loss
        
        if self.use_loss_weights and not self.ignore_loss:
            loss = loss * y_loss_weights

        if self.scalar_loss:
            return np.mean(loss)
        else:
            return np.mean(loss, axis=tuple(range(1,loss.ndim)))

class mixed_loss():
    def __init__(self, alpha=0.25, gamma=2., beta=1000., kappa=100., normalize=False, use_loss_weights=False):
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.kappa = kappa
        self.loss_focal = focal_loss(gamma=gamma, alpha=alpha, use_loss_weights=use_loss_weights)
        self.loss_dice = dice_coef_loss(use_loss_weights=use_loss_weights)
        self.loss_tp = tp_loss()
        self.__name__ = 'mixed_l'

    def __call__(self, y_true, y_pred):
        #return self.loss_focal(y_true, y_pred) + self.beta * self.loss_dice(y_true, y_pred) + self.kappa * self.loss_tp(y_true, y_pred)
        return self.beta * self.loss_focal(y_true, y_pred) + self.kappa * self.loss_dice(y_true, y_pred)

class mixed_loss_np():
    def __init__(self, alpha=0.25, gamma=2., beta=1000., kappa=100., normalize=False, use_loss_weights=False):
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.kappa = kappa
        self.loss_focal = focal_loss_np(gamma=gamma, alpha=alpha, use_loss_weights=use_loss_weights, scalar_loss=False)
        self.loss_dice = dice_coef_loss_np(use_loss_weights=use_loss_weights, scalar_loss=False)
        self.loss_tp = tp_loss()
        self.__name__ = 'mixed_l'

    def __call__(self, y_true, y_pred):
        #return self.loss_focal(y_true, y_pred) + self.beta * self.loss_dice(y_true, y_pred) + self.kappa * self.loss_tp(y_true, y_pred)
        return self.beta * self.loss_focal(y_true, y_pred) + self.kappa * self.loss_dice(y_true, y_pred)

class mixed_loss2():
    def __init__(self, alpha=0.25, gamma=2., beta=1000., kappa=100., normalize=False, use_loss_weights=False):
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.kappa = kappa
        self.loss_focal = focal_loss(gamma=gamma, alpha=alpha, use_loss_weights=use_loss_weights)
        self.loss_bce = bce(use_loss_weights=use_loss_weights)
        self.loss_tp = tp_loss()
        self.__name__ = 'mixed_l'

    def __call__(self, y_true, y_pred):
        return self.beta * self.loss_focal(y_true, y_pred) + self.kappa * self.loss_bce(y_true, y_pred)

class mixed_loss2_np():
    def __init__(self, alpha=0.25, gamma=2., beta=1000., kappa=100., normalize=False, use_loss_weights=False, scalar_loss=True):
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.kappa = kappa
        self.loss_focal = focal_loss_np(gamma=gamma, alpha=alpha, use_loss_weights=use_loss_weights, scalar_loss=scalar_loss)
        self.loss_bce = bce_np(use_loss_weights=use_loss_weights, scalar_loss=scalar_loss)
        self.__name__ = 'mixed_l'

    def __call__(self, y_true, y_pred):
        return self.beta * self.loss_focal(y_true, y_pred) + self.kappa * self.loss_bce(y_true, y_pred)

class true_positive_rate():
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.__name__ = 'tp'

    def __call__(self, y_true, y_pred):
        y_t_flat = K.flatten(y_true)
        y_p_flat = K.flatten(y_pred)
        return K.sum(y_t_flat * y_p_flat + self.eps)/K.sum(y_t_flat + self.eps)


class tp_loss():
    def __init__(self, eps=1e-6):
        self.tp = true_positive_rate(eps)
        self.__name__ = 'tp_l'
        
    def __call__(self, y_true, y_pred):
        return 1 - self.tp(y_true, y_pred)

def dice_tp_loss(y_true, y_pred, alpha=0.999):
    return alpha * dice_coef_loss(y_true, y_pred) + (1-alpha) * tp_loss(y_true, y_pred) 

thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

def f2(masks_true, masks_pred):
    if np.sum(masks_true) == 0:
        return float(np.sum(masks_pred) == 0)
    
    ious = []
    mp_idx_found = []
    for mt in masks_true:
        for mp_idx, mp in enumerate(masks_pred):
            if mp_idx not in mp_idx_found:
                cur_iou = iou_np(mt,mp)
                if cur_iou > 0.5:
                    ious.append(cur_iou)
                    mp_idx_found.append(mp_idx)
                    break
    f2_total = 0
    for th in thresholds:
        tp = sum([iou > th for iou in ious])
        fn = len(masks_true) - tp
        fp = len(masks_pred) - tp
        f2_total += (5*tp)/(5*tp + 4*fn + fp)    

    return f2_total/len(thresholds)

