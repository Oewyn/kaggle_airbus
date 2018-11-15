import pandas as pd
import os
from img_gen import StratifiedImgGen, ThreadedWorker, OHEMGen


class_empty = 0
class_1ship = 1
class_few_ship = 2
class_many_ship = 3

def classify(ships):
    if ships == 0:
        return class_empty
    elif ships == 1:
        return class_1ship
    else:
        return class_many_ship

class AirbusTrainValGen:

    def __init__(self,train_image_dir, batch_size, valid_batch_size, target_size, augment, crop, noship_detector=False, use_loss_weights=False, ship_grad_ratio=0.5, border_grad_ratio=0.75, online_hard_example_mining=False):
        masks = pd.read_csv(os.path.join('../data2/v2', 'train_ship_segmentations_v2.csv'))

        masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
        unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
        masks.drop(['ships'], axis=1, inplace=True)

        # some files are too small/corrupt
        exclude_list = ['6384c3e78.jpg'] #corrupted images
        unique_img_ids = unique_img_ids[~unique_img_ids['ImageId'].isin(exclude_list)]

        unique_img_ids['class'] = unique_img_ids['ships'].map(classify)
        unique_img_ids.drop(['ships'], axis=1, inplace=True)
        if not noship_detector:
            unique_img_ids = unique_img_ids.loc[unique_img_ids['class'] > class_empty]

        from sklearn.model_selection import train_test_split
        # Split into training/validation groups
        train_ids, valid_ids = train_test_split(unique_img_ids, 
                         test_size = 0.05, 
                         stratify = unique_img_ids['class'])
        train_df = pd.merge(masks, train_ids)
        valid_df = pd.merge(masks, valid_ids)

        '''
        self.train_gen = ThreadedWorker(
                StratifiedImgGen(train_df.groupby('class'),
                                 train_image_dir,
                                 batch_size,
                                 target_size=target_size,
                                 augment=augment,
                                 crop=crop,
                                 noship_detector=noship_detector,
                                 use_loss_weights=use_loss_weights,
                                 use_threads=False,
                                 ship_grad_ratio=ship_grad_ratio,
                                 border_grad_ratio=border_grad_ratio,
                                 online_hard_example_mining=online_hard_example_mining),
                queue_size=8)
        '''

        self.train_gen = \
                StratifiedImgGen(train_df.groupby('class'),
                                 train_image_dir,
                                 batch_size,
                                 target_size=target_size,
                                 augment=augment,
                                 crop=crop,
                                 noship_detector=noship_detector,
                                 use_loss_weights=use_loss_weights,
                                 use_threads=False,
                                 ship_grad_ratio=ship_grad_ratio,
                                 border_grad_ratio=border_grad_ratio,
                                 online_hard_example_mining=online_hard_example_mining)

        self.ohem_input_queue = []

        if online_hard_example_mining:
            self.train_ids = self.train_gen.gen_img_ids
            self.ohem_gen = OHEMGen(in_df=train_df,
                                    image_dir=train_image_dir,
                                    target_size=target_size,
                                    crop=crop,
                                    input_queue=self.ohem_input_queue,
                                    noship_detector=noship_detector,
                                    use_loss_weights=use_loss_weights,
                                    ship_grad_ratio=ship_grad_ratio,
                                    border_grad_ratio=border_grad_ratio,
                                    report_imageid_queue=self.train_gen.ohem_queues[-1])
            self.train_gen.add_gen(self.ohem_gen)

        self.valid_gen = StratifiedImgGen(valid_df.groupby('class'),
                                          train_image_dir,
                                          valid_batch_size,
                                          target_size=target_size,
                                          augment=False,
                                          crop=crop,
                                          noship_detector=noship_detector,
                                          use_loss_weights=use_loss_weights,
                                          ship_grad_ratio=ship_grad_ratio,
                                          border_grad_ratio=border_grad_ratio,
                                          online_hard_example_mining=online_hard_example_mining)
