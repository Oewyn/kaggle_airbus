from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, SeparableConv2D, Activation, GlobalMaxPooling2D, Reshape, Flatten
from keras.layers import Input, add, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import Constant

import keras.backend as K
import tensorflow as tf
import math

from losses import *

def encoder(x, filters=44, n_block=3, kernel_size=(3,3), activation='relu'):
    skip = []
    for i in range(n_block):
        layer_name = 'encode{}/'.format(i)
        x = Conv2D(filters * (2 ** i), kernel_size, activation=activation, padding='same', name=layer_name+'Conv2D_A')(x)
        x = Conv2D(filters * (2 ** i), kernel_size, activation=activation, padding='same', name=layer_name+'Conv2D_B')(x)
        #x = BatchNormalization(name=layer_name+'BatchNorm')(x)
        skip.append(x)
        if i != n_block-1:
            x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name=layer_name+'MaxPool2D')(x)
    return x, skip
         
def bottleneck(x, filters_bottleneck, mode='cascade', depth=6, kernel_size=(3,3), activation='relu'):
    dilated_layers = []
    layer_name = 'btlneck_{}/'.format(mode)
    for i in range(depth):
        if mode == 'cascade':
            x = Conv2D(filters_bottleneck, kernel_size, activation=activation, padding='same', dilation_rate=2**i, name=layer_name+'Conv2D_{}'.format(i))(x)
            dilated_layers.append(x)
        elif mode == 'parallel':
            dilated_layers.append(Conv2D(filters_bottleneck, kernel_size, activation=activation, padding='same', dilation_rate=2**i, name=layer_name+'Conv2D_{}'.format(i))(x))
    return add(dilated_layers)


def decoder(x, skip, filters, n_block=3, kernel_size=(3,3), activation='relu'):
    for i in reversed(range(n_block)):
        layer_name = 'decode{}/'.format(i)
        if i != n_block-1:
            x = UpSampling2D(size=(2,2), name=layer_name+'Upsample')(x)
        x = Conv2D(filters * 2**i, kernel_size, activation=activation, padding='same', name=layer_name+'Conv2D_A')(x)
        x = concatenate([skip[i], x], name=layer_name+'Concat')
        x = Conv2D(filters * 2**i, kernel_size, activation=activation, padding='same', name=layer_name+'Conv2D_B')(x)
        x = Conv2D(filters * 2**i, kernel_size, activation=activation, padding='same', name=layer_name+'Conv2D_C')(x)
        #x = BatchNormalization(name=layer_name+'BatchNorm')(x)
    return x

def get_unet(input_shape=(768,768,3),
    mode='cascade',
    filters=8,
    n_block=3,
    lr=1e-4,
    loss=focal_loss(),
    n_class=1,
    prob_true=0.01,
    use_loss_weights=False
):
    inputs = Input(input_shape)
    x = BatchNormalization(name='BatchNorm')(inputs)

    enc, skip = encoder(x, filters, n_block)
    bottle = enc
    dec = decoder(bottle, skip, filters, n_block)
    #classify = Conv2D(n_class, (1,1), activation='sigmoid')(dec)
    bias_value = -math.log((1-prob_true)/prob_true)
    print('classify bias = {}'.format(bias_value))
    classify = Conv2D(n_class, (1,1), activation='sigmoid', bias_initializer=Constant(value=bias_value))(dec)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=Adam(lr=lr, decay=0.0), loss=loss, metrics=[focal_loss(alpha=.125, gamma=1.5, normalize=True, use_loss_weights=use_loss_weights), dice_coef(use_loss_weights=use_loss_weights)])

    return model

def fire(x, layer_name='fire/', sq_filters=16, ex_filters=32, activation='relu'):
    x = Conv2D(sq_filters, (1,1), activation=activation, padding='same', name=layer_name+'sq_1x1')(x)
    e1 = Conv2D(ex_filters, (1,1), activation=activation, padding='same', name=layer_name+'exp1x1')(x)
    e3 = Conv2D(ex_filters, (3,3), activation=activation, padding='same', name=layer_name+'exp3x3')(x)
    x = concatenate([e1, e3], name=layer_name+'concat')
    return x

def separable(x, layer_name='sep_conv/', sq_filters=16, ex_filters=32, activation='relu'):
    x = BatchNormalization(name=layer_name+'BatchNorm')(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(filters=sq_filters, kernel_size=(3,3), activation=activation, padding='same', name=layer_name+'sep_3x3', depth_multiplier=2, use_bias=False)(x)
    return x

def encoder_sq(x, filters=16, n_block=3, activation='relu', batnorm=False, expansion_schedule=None, fire=True):
    skip = []
    expansions = 0
    for i in range(n_block):
        expanded_this_layer = False
        if expansion_schedule == None or expansion_schedule[i] == True:
            expansions += 1
            expanded_this_layer = True
        layer_name = 'encode{}/'.format(i)
        sq_filters = int(filters * (2 ** expansions))
        if fire:
            x = fire(x, layer_name=layer_name+'fire/', sq_filters=sq_filters, ex_filters=2*sq_filters)
        else:
            x = separable(x, layer_name=layer_name+'sep/', sq_filters=sq_filters, ex_filters=2*sq_filters)
        if batnorm:
            x = BatchNormalization(name=layer_name+'BatchNorm')(x)
        skip.append(x)
        if expanded_this_layer:
            x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name=layer_name+'MaxPool2D')(x)
    return x, skip

def decoder_sq(x, skip, filters=16, n_block=3, activation='relu', batnorm=False, expansion_schedule=None, fire=True):

    expansions = sum(expansion_schedule[0:n_block])
    for i in reversed(range(n_block)):
        contracted_this_layer = False
        if expansion_schedule == None or expansion_schedule[i] == True:
            expansions -= 1
            contracted_this_layer = True
        layer_name = 'decode{}/'.format(i)
        sq_filters = int(filters * (2 ** expansions))
        if contracted_this_layer:
            x = UpSampling2D(size=(2,2), name=layer_name+'Upsample')(x)
            x = Conv2D(filters * 2**expansions, (3,3), activation=activation, padding='same', name=layer_name+'Conv2D_A')(x)
            x = concatenate([skip[i], x], name=layer_name+'Concat')
        if fire:
            x = fire(x, layer_name=layer_name+'fire/', sq_filters=sq_filters, ex_filters=2*sq_filters)
        else:
            x = separable(x, layer_name=layer_name+'sep/', sq_filters=sq_filters, ex_filters=2*sq_filters)
        if batnorm:
            x = BatchNormalization(name=layer_name+'BatchNorm')(x)
    return x

def get_unet_sq(input_shape=(768,768,3),
    mode='cascade',
    filters=8,
    n_block=3,
    loss=dice_coef_loss(),
    n_class=1,
    prob_true=0.25,
    optimizer=Adam(lr=1e-4, decay=0.0),
    batnorm=False,
    expansion_schedule=None,
    fire=True,
    use_loss_weights=False
):
    inputs = Input(input_shape)
    x = BatchNormalization(name='BatchNorm')(inputs)
    x = Conv2D(filters*4, (3,3), activation='relu', padding='same', name='Conv2D1')(x)

    enc, skip = encoder_sq(x, filters, n_block, batnorm=batnorm, expansion_schedule=expansion_schedule, fire=fire)
    bottle = enc
    dec = decoder_sq(bottle, skip, filters, n_block, batnorm=batnorm, expansion_schedule=expansion_schedule, fire=fire)
    bias_value = -math.log((1-prob_true)/prob_true)
    classify = Conv2D(n_class, (1,1), activation='sigmoid', bias_initializer=Constant(value=bias_value))(dec)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=optimizer, loss=loss, metrics=[focal_loss(use_loss_weights=use_loss_weights, ignore_loss=True), bce(use_loss_weights=use_loss_weights), dice_coef_loss(use_loss_weights=use_loss_weights, ignore_loss=True), dice_coef(use_loss_weights=use_loss_weights, ignore_loss=True)])

    return model

def get_resnet34(input_shape=(768,768,3),
    loss=dice_coef_loss(),
    n_class=1,
    optimizer=Adam(lr=1e-4, decay=0.0),
    use_loss_weights=False
):
    from unet_model import Unet
    model = Unet(backbone_name='resnet34', input_shape=input_shape, encoder_weights='imagenet')

    model.compile(optimizer=optimizer, loss=loss, metrics=[focal_loss(use_loss_weights=use_loss_weights, ignore_loss=True), bce(use_loss_weights=use_loss_weights), dice_coef_loss(use_loss_weights=use_loss_weights, ignore_loss=True), dice_coef(use_loss_weights=use_loss_weights, ignore_loss=True)])

    return model

def get_resnet34_backbone(input_shape=(768,768,3),
    loss=dice_coef_loss(),
    n_class=1,
    optimizer=Adam(lr=1e-4, decay=0.0),
    use_loss_weights=False
):
    from resnet_model import ResNet34
    model = ResNet34(input_shape=input_shape, include_top=False, classes=1, class_detector_top=True)

    from keras import metrics
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics.binary_accuracy])

    return model

def get_unet_sq_backbone(input_shape=(768,768,3),
    mode='cascade',
    filters=8,
    n_block=3,
    loss=dice_coef_loss(),
    n_class=1,
    prob_true=0.25,
    optimizer=Adam(lr=1e-4, decay=0.0),
    batnorm=False,
    expansion_schedule=None,
    fire=True
):
    inputs = Input(input_shape)
    x = BatchNormalization(name='BatchNorm')(inputs)
    x = Conv2D(filters*4, (3,3), activation='relu', padding='same', name='Conv2D1')(x)

    enc, skip = encoder_sq(x, filters, n_block, batnorm=batnorm, expansion_schedule=expansion_schedule, fire=fire)
    pooled = GlobalMaxPooling2D()(enc)
    pooled = Reshape((1,1,-1))(pooled)
    
    classify = Conv2D(n_class, (1,1), name='classifier', activation='sigmoid')(pooled)
    classify = Flatten()(classify)

    model = Model(inputs=inputs, outputs=classify)

    from keras import metrics
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics.binary_accuracy])

    return model
