x1 = 1
y1 = 0
x2 = 3
y2 = 2

import math
import cv2
import numpy as np

from skimage.morphology import disk
from skimage.measure import label, regionprops
from rle import rle_decode

def masks_as_image(in_mask_list, shape=(768,768)):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros(shape, dtype = np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask, shape)
    return np.expand_dims(all_masks, -1)

def masks_as_loss_weights(in_mask_list, shape=(768,768), ship_grad_ratio=0.1, border_grad_ratio=0.75):
    total_grad = shape[0] * shape[1]
    ship_grad = total_grad * ship_grad_ratio
    per_pixel_background_grad = (total_grad - ship_grad) / total_grad
    num_ship = len(in_mask_list)
    per_ship_border_grad = ship_grad * border_grad_ratio / num_ship
    per_ship_inner_grad = ship_grad * (1 - border_grad_ratio) / num_ship

    #print('total_grad: {} ship_grad: {}, per_pixel_background_grad: {}'.format(total_grad, ship_grad, per_pixel_background_grad))
    #print('num_ship: {} per_ship_border_grad: {}, per_ship_inner_grad: {}'.format(num_ship, per_ship_border_grad, per_ship_inner_grad))

    structure = disk(3)
    # Take the individual ship masks and create  and create a single mask array for all ships
    all_masks = np.full(shape=shape, fill_value=per_pixel_background_grad, dtype=np.float32)
    for mask in in_mask_list:
        if isinstance(mask, str):
            base_mask = rle_decode(mask, shape)
            inner_mask = cv2.erode(base_mask, structure)
            border_mask = cv2.dilate(base_mask, structure) - inner_mask

            # allocate gradient to masks
            inner_mask_pixels = np.sum(inner_mask)
            border_mask_pixels = np.sum(border_mask)
            if inner_mask_pixels > 0:
                inner_per_pix = per_ship_inner_grad / inner_mask_pixels
                border_per_pix = per_ship_border_grad / border_mask_pixels
                inner_mask = inner_per_pix * inner_mask.astype(np.float32)
                border_mask = border_per_pix * border_mask.astype(np.float32)
            else:
                #print('border mask taking all ship gradient')
                border_per_pix = (per_ship_inner_grad + per_ship_border_grad) / border_mask_pixels
                border_mask = border_per_pix * border_mask.astype(np.float32)

            all_masks += inner_mask + border_mask
    #print('total grad: {}, should be: {}'.format(np.sum(all_masks), total_grad))
    #print('shape: {} dtype: {}'.format(all_masks.shape, all_masks.dtype))
    return np.expand_dims(all_masks, -1)

def post_process_prediction(img, structure_size=3, iterations=1):
    structure = disk(structure_size)
    new_mask = img.copy()
    for i in range(iterations):
        new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, structure)
        new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_OPEN, structure)

    return np.expand_dims(new_mask, -1)

def masks_as_color(in_mask_list, shape=(768,768)):
    # Take the individual ship masks and create a color mask array for each ships
    all_masks = np.zeros(shape=shape, dtype = np.float)
    scale = lambda x: (len(in_mask_list)+x+1) / (len(in_mask_list)*2) ## scale the heatmap image to shift 
    for i,mask in enumerate(in_mask_list):
        if isinstance(mask, str):
            all_masks[:,:] += scale(i) * rle_decode(mask)
    return all_masks

def expand_bbox(bbox, target_size):
    img_size = 768
    target_pow2 = int(math.log(target_size, 2))
    
    if bbox == []:
        # TODO: FIX
        bbox = [0, 0, 64, 64]

    width = bbox[x2] - bbox[x1]
    height = bbox[y2] - bbox[y1]
    max_dim = max(width, height)
    
    oversample = 1
    if max_dim > target_size:
        oversample = math.ceil(math.log(max_dim, 2) - target_pow2 + 1)
        
    crop_size = min(img_size, 2 ** (target_pow2 + oversample - 1))
    crop_half = int(crop_size / 2)
    c_x = int((bbox[x2] + bbox[x1])/2)
    c_y = int((bbox[y2] + bbox[y1])/2)
    
    c_x = max(crop_half, c_x)
    c_x = min((img_size-1) - crop_half, c_x)
    c_y = max(crop_half, c_y)
    c_y = min((img_size-1) - crop_half, c_y)
    
    new_bbox = [0,0,0,0]
    new_bbox[x1] = c_x - crop_half
    new_bbox[x2] = c_x + crop_half
    new_bbox[y1] = c_y - crop_half
    new_bbox[y2] = c_y + crop_half
        
    return new_bbox, crop_size

def grab_ship_img_mask(img, rle, target_size, crop=True, use_loss_weights=False, ship_grad_ratio=0.5, border_grad_ratio=0.75):
    mask = masks_as_image(rle, (img.shape[0], img.shape[1]))

    if use_loss_weights:
        loss_weights = masks_as_loss_weights(rle, (img.shape[0], img.shape[1]), ship_grad_ratio=ship_grad_ratio, border_grad_ratio=border_grad_ratio)
    
    if crop:
        lbl = label(mask) 
        props = regionprops(lbl)
        orig_bbox = []
            
        for prop in props:
            orig_bbox = prop.bbox
            break;
            
        img_bbox, crop_size = expand_bbox(orig_bbox, target_size)
            
        y_s = slice(img_bbox[y1],img_bbox[y2])
        x_s = slice(img_bbox[x1],img_bbox[x2])
    else:
        y_s = slice(0, img.shape[0])
        x_s = slice(0, img.shape[1])
        crop_size = img.shape[0]
    
    if crop_size == target_size:
        if not use_loss_weights:
            return img[y_s, x_s, :], mask[y_s, x_s, :]
        else:
            return img[y_s, x_s, :], (mask[y_s, x_s, :], loss_weights[y_s, x_s, :])
    else:
        resize_img = cv2.resize(src=img[y_s, x_s, :], dsize=(target_size, target_size), interpolation = cv2.INTER_CUBIC)
        resize_mask = cv2.resize(src=mask[y_s, x_s, :], dsize=(target_size, target_size), interpolation = cv2.INTER_CUBIC)
    if not use_loss_weights:
        return resize_img, np.expand_dims(resize_mask, axis=2)
    else:
        resize_loss_weights = cv2.resize(src=loss_weights[y_s, x_s, :], dsize=(target_size, target_size), interpolation = cv2.INTER_CUBIC)
        return resize_img, (np.expand_dims(resize_mask, axis=2), np.expand_dims(resize_loss_weights, axis=2))

def tta_batch(img):
    flips = ['H', 'V']
    rotations = [0, 90, 180, 270]
    return [ rotate_batch_img(img, degrees=deg) for deg in rotations ] + [ flip_batch_img(img, orientation=flip) for flip in flips ]

def tta_batch_reverse(img_list):
    flips = ['H', 'V']
    rotations = [0, -90, -180, -270]
    return [rotate_batch_img(i, degrees=deg) for i, deg in zip(img_list[:3], rotations) ] + [ flip_batch_img(i, orientation=flip) for i, flip in zip(img_list[4:], flips) ]

def rotate_batch_img(img, degrees=90):
    H = img.shape[1]
    W = img.shape[2]
    rot_matrix = cv2.getRotationMatrix2D((H/2, W/2), degrees, scale=1)
    return np.stack([cv2.warpAffine(i, rot_matrix, (H, W)) for i in img[0:,...]], 0)

def flip_batch_img(img, orientation='H'):
    if orientation == 'H':
        axis = 0
    else:
        axis = 1
    return np.stack([cv2.flip(i, axis) for i in img[0:,...]], 0)
