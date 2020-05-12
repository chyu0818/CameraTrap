import os
from torch.utils.data import Dataset
import random
import numpy as np
import torch
import json
from PIL import Image
# 1 animal 2 person for bbox human id 75
# data augment During training, images were randomly cropped and perturbed in brightness, saturation, hue,and contrast
# with open('iwildcam2020_train_annotations.json') as f:
#     data = json.load(f)
# print(len(data['annotations']))
# print(len(data['images']))
# print(data['annotations'][1000])
#
# with open('iwildcam2020_megadetector_results.json') as f:
#     data = json.load(f)
# print(len(data['images']))
# hi = np.load('X_train.npz')
# print(hi.files)
HUMAN_CATEGORY_ID = 75
class CameraTrapDataset(Dataset):
    def __init__(self, im_fp, file_lst_fn, annotations_fn, bbox_fn, seed=1, transform=None):
        '''
        im_fp: filepath for images
        file_lst_fn: filename for list of images
        annotations_fn: filename for annotations
        bbox_fn: filename for detections
        '''
        self.im = None
        self.im_lst = []   # List of cropped images from bounding boxes.
        self.target_lst = []   # List of target categories.
        self.id_lst = []   # List of image ids corresponding to each bounding box.
        self.conf = []   # List of confidence values.
        self.transform = transform
        random.seed(seed)

        # Randomize files.
        file_lst = np.load(file_lst_fn)['arr_0'][:1]
        np.random.shuffle(file_lst)
        print('Total number of files: ', len(file_lst))
        
        empty_count = 0
        empty_w_bbox_count = 0
        nonempty_no_bbox_count = 0
        for f in file_lst:
            # Get detections and category id from dictionary.
            detect = []
            category_id = 1
            assert(type(category_id) is int)

            # Read in image.
            im = Image.open(os.path.join(im_fp,'{}.jpg'.format(f)))
            print(im)
            (n_rows, n_cols, n_channels) = np.shape(im)
            print(np.shape(im))
            self.im = self.transform(im)


            self.id_lst.append(1)
            self.conf.append(0.9)

            self.im_lst.append(im)
        print(len(self.im_lst))


    def __len__(self):
        return len(self.im_lst)

    def __getitem__(self, idx):
        return {'image':self.im, 'target':1, 'id':1}

