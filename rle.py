import numpy as np
from skimage.morphology import label
from skimage.morphology import binary_erosion, square, disk

import cv2

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# take unified segmentation mask, split into contiguous objects via label, then generate
# RLE for each individual label
def multi_rle_encode(img):
    labels = label(img[:,:,0], connectivity=2)
    return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]

def instance_segment_mask(img, smallest_size=50):
    labels = label(img, connectivity=2)
    inst_segs = []
    for l, len_l in zip(*np.unique(labels[labels>0], return_counts=True)):
        if len_l >= smallest_size:
            inst_segs += [labels == l]
    return inst_segs

from losses import iou_np
def instance_segment_box(img, smallest_size=50, min_iou=0.6):
    cvt_img = (255*img).astype('uint8').copy()
    _, contours, _ = cv2.findContours(cvt_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    inst_segs = []
    for c in contours:
        if len(c) < smallest_size:
            continue
        e_center, e_size, e_rotation = cv2.fitEllipse(c)
        e_major, e_minor = max(e_size), min(e_size)
        r_center, r_size, r_rotation = cv2.minAreaRect(c)
        r_major, r_minor = max(r_size), min(r_size)

        box = cv2.boxPoints((e_center, (e_minor, r_major), e_rotation))
        box = np.int0(box)
        box_mask = np.zeros((img.shape[0], img.shape[1], 1), dtype='uint8')
        c_mask = np.zeros((img.shape[0], img.shape[1], 1), dtype='uint8')
        inst_segs += [np.zeros((img.shape[0], img.shape[1], 1), dtype='uint8')]
        cv2.fillPoly(box_mask,[box],(255))
        cv2.fillPoly(c_mask, pts=[c], color=(255))
        box_mask = box_mask.astype('float32')/255.
        c_mask = c_mask.astype('float32')/255.
        iou = iou_np(np.expand_dims(box_mask, axis=0), np.expand_dims(c_mask, axis=0))

        # iou check
        if iou > min_iou:
            inst_segs += [box_mask]

    return inst_segs

def rle_decode(mask_rle, shape=(768,768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def rle_find_center(mask_rle, shape=(768,768)):
    img = rle_decode(mask_rle, shape)
    orig_pixels = img.sum()
    target_pixels = orig_pixels/4.
    big_structure = disk(3)
    little_structure = square(2)
    use_big_structure = True

    structure = big_structure
    while img.sum() > target_pixels:
        tmp_img = binary_erosion(img, structure).astype(np.float32)
        #print(tmp_img.sum())
        if tmp_img.sum() > target_pixels:
            img = tmp_img
        else:
            if not use_big_structure:
                break
            else:
                structure = little_structure
                use_big_structure = False

    #print('lit pixels: before {} after {}'.format(orig_pixels, img.sum()))
    return rle_encode(img.T) # transpose to make RLE match training data which is transposed

