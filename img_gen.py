import numpy as np
import cv2
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

from img_preprocess import *

class StratifiedImgGen:
    def __init__(self, groups, image_dir, batch_size, target_size=None, augment=True, crop=False, use_threads=False, noship_detector=False, use_loss_weights=False, ship_grad_ratio=0.5, border_grad_ratio=0.75, online_hard_example_mining=False):
        self.batch_size = batch_size
        self.target_size = target_size
        self.generators = []
        self.crop = crop
        self.use_loss_weights = use_loss_weights
        self.gen_img_ids = []
        self.ohem = online_hard_example_mining
        if online_hard_example_mining:
            self.ohem_queues = [list()]
        else:
            self.ohem_queues = [None]
        for ships, all_batches in groups:
            if len(all_batches) > 0:
                if augment:
                    if use_threads:
                        self.add_gen(
                                ThreadedWorker(
                                    create_aug_gen(
                                        self.make_image_gen((ships, all_batches),
                                                             noship_detector=noship_detector,
                                                             use_loss_weights=use_loss_weights,
                                                             ship_grad_ratio=ship_grad_ratio,
                                                             border_grad_ratio=border_grad_ratio, 
                                                             report_imageid_queue=self.ohem_queues[-1]),
                                        noship_detector=noship_detector, 
                                        use_loss_weights=use_loss_weights)))
                    else:
                        self.add_gen(
                                create_aug_gen(
                                    self.make_image_gen((ships, all_batches),
                                                        noship_detector=noship_detector,
                                                        use_loss_weights=use_loss_weights,
                                                        ship_grad_ratio=ship_grad_ratio,
                                                        border_grad_ratio=border_grad_ratio,
                                                        report_imageid_queue=self.ohem_queues[-1]),
                                    noship_detector=noship_detector,
                                    use_loss_weights=use_loss_weights))
                else:
                    self.add_gen(
                            self.make_image_gen((ships, all_batches),
                                                noship_detector=noship_detector,
                                                use_loss_weights=use_loss_weights,
                                                ship_grad_ratio=ship_grad_ratio,
                                                border_grad_ratio=border_grad_ratio,
                                                report_imageid_queue=self.ohem_queues[-1]))
        self.image_dir = image_dir

    def add_gen(self, generator):
        self.generators.append(generator)
        # always prepare next queue entry ahead of time so we can use [-1] index
        if self.ohem:
            self.ohem_queues.append(list())


    def __next__(self):
        out_rgb = []
        out_mask = []
        out_loss_weights = []
        strata = np.random.randint(0, len(self.generators), self.batch_size)
        for b in strata:
            c_img, c_mask = next(self.generators[b])
            if self.use_loss_weights:
                c_mask, c_loss_weights = c_mask
                out_loss_weights += [c_loss_weights]

            out_rgb += [c_img]
            out_mask += [c_mask]
            if self.ohem:
                self.gen_img_ids.append(self.ohem_queues[b].pop(0))
        if not self.use_loss_weights:
            return np.concatenate(out_rgb, axis=0), np.concatenate(out_mask, axis=0)
        else:
            return np.concatenate(out_rgb, axis=0), np.concatenate([np.concatenate(out_mask, axis=0), np.concatenate(out_loss_weights, axis=0)], axis=-1)

    def make_image_gen(self, in_df, noship_detector=False, use_loss_weights=False, ship_grad_ratio=0.5, border_grad_ratio=0.75, report_imageid_queue=None):
        name, all_batches = in_df
        all_batches = list(all_batches.groupby('ImageId'))
        indices = [k for k in range(0, len(all_batches))]
        out_rgb = []
        out_mask = []
        while True:
            #print('ships: {} shuffling img_gen group of len: {}'.format(name, len(all_batches)))
            np.random.shuffle(indices)
            for idx in indices:
                c_img_id, c_masks = all_batches[idx]
                if report_imageid_queue is not None:
                    report_imageid_queue.append(c_img_id)
                input_img =  cv2.imread(self.image_dir+'/' + c_img_id)
                c_img = {}
                c_mask = {}
                if self.target_size != None:
                    ### Small image Crop
                    c_img, c_mask = grab_ship_img_mask(input_img, c_masks['EncodedPixels'].values, self.target_size, crop=self.crop, use_loss_weights=use_loss_weights, ship_grad_ratio=ship_grad_ratio, border_grad_ratio=border_grad_ratio)
                    if use_loss_weights:
                        c_mask, c_loss_weights = c_mask
                else:
                    ### FULL IMAGE NO CROP
                    c_img = input_img
                    c_mask = masks_as_image(c_masks['EncodedPixels'].values, (c_img.shape[0],c_img.shape[1]))
                    if use_loss_weights:
                        c_loss_weights = masks_as_loss_weights(c_masks['EncodedPixels'].values, (c_img.shape[0],c_img.shape[1]))
                yield_x = preprocess_img(c_img)
                if not noship_detector:
                    yield_y = c_mask
                    if use_loss_weights:
                        yield_lw = c_loss_weights
                else:
                    yield_y = np.any(c_mask).astype(np.float32)

                if len(yield_x.shape) == 3 and len(c_mask.shape) == 3:
                    yield_x = np.expand_dims(yield_x, axis=0)
                    yield_y = np.expand_dims(yield_y, axis=0)
                    if use_loss_weights:
                        yield_lw = np.expand_dims(yield_lw, axis=0)
                if not use_loss_weights:
                    yield yield_x, yield_y
                else:
                    yield yield_x, (yield_y, yield_lw)

def preprocess_img(img_data):
    return (img_data / 255.0) - 0.5

def make_test_gen(img_list, image_dir, batch_size, target_size=768):
    out_rgb = []
    out_img_name = []
    for c_img_id in img_list:
        input_img =  cv2.imread(image_dir+'/' + c_img_id)
        c_img = cv2.imread(image_dir+'/' + c_img_id)
        if c_img.shape[0] != target_size:
            c_img = cv2.resize(src=c_img, dsize=(target_size, target_size), interpolation = cv2.INTER_CUBIC)
        out_rgb += [preprocess_img(c_img)]
        out_img_name += [c_img_id]
        if len(out_rgb)>=batch_size:
            yield (np.stack(out_rgb, 0)), out_img_name
            out_rgb = []
            out_img_name = []
    if len(out_rgb) > 0:
        yield (np.stack(out_rgb, 0)), out_img_name


def create_aug_gen(in_gen, seed = None, noship_detector=False, use_loss_weights=False):
    dg_args = dict(featurewise_center = False, 
                      samplewise_center = False,
                      rotation_range = 15, 
                      width_shift_range = 0.1, 
                      height_shift_range = 0.1, 
                      shear_range = 0.01,
                      zoom_range = [0.5, 1.5],  
                      horizontal_flip = True, 
                      vertical_flip = True,
                      fill_mode = 'reflect',
                      data_format = 'channels_last')

    image_gen = ImageDataGenerator(**dg_args)
    label_gen = ImageDataGenerator(**dg_args)
    if use_loss_weights:
        loss_weight_gen = ImageDataGenerator(**dg_args)

    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        g_x = image_gen.flow(255*in_x, 
                             batch_size = in_x.shape[0], 
                             seed = seed, 
                             shuffle=True)
        if use_loss_weights:
            in_y, in_lw = in_y

        if not noship_detector:
            g_y = label_gen.flow(in_y, 
                                 batch_size = in_x.shape[0], 
                                 seed = seed, 
                                 shuffle=True)

            if use_loss_weights:
                g_lw = loss_weight_gen.flow(in_lw, 
                                     batch_size = in_x.shape[0], 
                                     seed = seed, 
                                     shuffle=True)
                yield next(g_x)/255.0, (next(g_y), next(g_lw))
            else:
                yield next(g_x)/255.0, next(g_y)
        else:
            yield next(g_x)/255.0, np.expand_dims(in_y, axis=0)

class OHEMGen:
    def __init__(self, in_df, image_dir, target_size, crop=False, input_queue=None, noship_detector=False, use_loss_weights=False, ship_grad_ratio=0.5, border_grad_ratio=0.75, report_imageid_queue=None):
        self.ohem_queue = ['410e65c96.jpg']
        self.report_imageid_queue = report_imageid_queue
        self.input_queue = input_queue
        if input_queue is None:
            self.input_queue = []
        self.noship_detector = noship_detector
        self.use_loss_weights = use_loss_weights
        self.ship_grad_ratio = ship_grad_ratio
        self.border_grad_ratio = border_grad_ratio
        self.image_dir = image_dir
        self.target_size = target_size
        self.crop = crop

        self.image_mask_dict = {image_id: masks['EncodedPixels'].values for image_id, masks in list(in_df.groupby('ImageId'))}

        self.indices = [0]

    def __iter__(self):
        return self

    def __next__(self):
        idx = self.indices.pop(0)

        c_img_id = self.ohem_queue[idx]
        c_masks = self.image_mask_dict[c_img_id]
        if self.report_imageid_queue is not None:
            self.report_imageid_queue.append(c_img_id)
        input_img =  cv2.imread(self.image_dir+'/' + c_img_id)
        c_img = {}
        c_mask = {}
        if self.target_size != None:
            ### Small image Crop
            c_img, c_mask = grab_ship_img_mask(input_img, c_masks, self.target_size, crop=self.crop, use_loss_weights=self.use_loss_weights, ship_grad_ratio=self.ship_grad_ratio, border_grad_ratio=self.border_grad_ratio)
            if self.use_loss_weights:
                c_mask, c_loss_weights = c_mask
        else:
            ### FULL IMAGE NO CROP
            c_img = input_img
            c_mask = masks_as_image(c_masks.values, (c_img.shape[0],c_img.shape[1]))
            if self.use_loss_weights:
                c_loss_weights = masks_as_loss_weights(c_masks, (c_img.shape[0],c_img.shape[1]))
        x = preprocess_img(c_img)
        if not self.noship_detector:
            y = c_mask
            if self.use_loss_weights:
                y_weights = c_loss_weights
        else:
            y = np.any(c_mask).astype(np.float32)

        if len(x.shape) == 3 and len(c_mask.shape) == 3:
            x = np.expand_dims(x, axis=0)
            y = np.expand_dims(y, axis=0)
            if self.use_loss_weights:
                y_weights = np.expand_dims(y_weights, axis=0)

        # grab new set of hard examples if available, then repopulate shuffled indices
        if len(self.indices) is 0:
            if len(self.input_queue) > 0:
                self.ohem_queue = self.input_queue.copy()
                self.input_queue.clear()

            self.indices = [i for i in range(0, len(self.ohem_queue))]
            np.random.shuffle(self.indices)

        if not self.use_loss_weights:
            #print('OHEMGen returning first value with shape {} {}'.format(x.shape, y.shape))
            return x, y
        else:
            return x, (y, y_weights)


import threading
from collections import deque

class ThreadedWorker(threading.Thread):
    def __init__(self, generator, queue_size=4):
        threading.Thread.__init__(self)
        self.generator = generator
        self.queue = deque(maxlen=queue_size)
        self.q_pop = threading.Condition()
        self.q_push = threading.Condition()
        self.daemon = True
        self.start()
        #self.detach()

    def __next__(self):
        if len(self.queue) == 0:
            with self.q_push:
                self.q_push.wait()

        result = self.queue.pop()
        with self.q_pop:
            self.q_pop.notifyAll()
        return result

    def run(self):
        while True:
            while len(self.queue) < self.queue.maxlen:
                self.queue.appendleft(next(self.generator))
                with self.q_push:
                    self.q_push.notifyAll()
            with self.q_pop:
                self.q_pop.wait()
