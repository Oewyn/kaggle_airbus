import pandas as pd
import numpy as np
from tqdm import tqdm


test_image_dir = '../data2/v2/test/'

resnet_seg = pd.read_csv('submissions/resnet34_768_ohem_best_loss_th_05_so_60_submission.csv')
resnet_noship = pd.read_csv('csvs/768_resnet_noship_ohem_dice_prob.csv')

resnet_seg_none = resnet_seg.loc[resnet_seg['EncodedPixels'].isnull()]

#print(resnet_seg_none.sample(10))

resnet_seg_none_image = set(resnet_seg_none['ImageId'])
#print(resnet_seg_none_image)

summary_probs = pd.read_csv('csvs/summary_noship_prob.csv')

test_prob_none_img = {}
steps = 20

total = len(summary_probs)
print('total = {}'.format(total))

for i in range(0, steps):
    percent = (i+1) * 1./(steps)
    test_prob_none_img['{:.2f}'.format(percent)] = set(summary_probs.loc[summary_probs['Average'] < (percent)]['ImageId'])
    num_pass = len(summary_probs.loc[summary_probs['Average'] > percent])
    print('threshold[{:.2f}] = {} ({:.2f})'.format(percent, num_pass, num_pass/total))

for test_name, test_set in test_prob_none_img.items():

    intersection = len(test_set.intersection(resnet_seg_none_image))
    union = len(test_set.union(resnet_seg_none_image))

    print('threshold: {} intersection = {} union = {}, IoU = {:.2f}'.format(test_name, intersection, union, intersection / union))

'''

'''
