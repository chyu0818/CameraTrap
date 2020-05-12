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
        self.im_lst = []   # List of cropped images from bounding boxes.
        self.target_lst = []   # List of target categories.
        self.id_lst = []   # List of image ids corresponding to each bounding box.
        self.conf = []   # List of confidence values.
        self.transform = transform
        random.seed(seed)

        # Randomize files.
        file_lst = np.load(file_lst_fn)['arr_0']
        np.random.shuffle(file_lst)
        print('Total number of files: ', len(file_lst))

        # Read in json files.
        with open(annotations_fn) as f:
            annotations = json.load(f)['annotations']
        with open(bbox_fn) as f1:
            detections = json.load(f1)['images']

        # Create dictionary by image id.
        im_dict = self.create_im_dict(annotations, detections)

        empty_count = 0
        empty_w_bbox_count = 0
        nonempty_no_bbox_count = 0
        for f in file_lst:
            # Get detections and category id from dictionary.
            detect = im_dict[f]['detections']
            category_id = im_dict[f]['category_id']
            assert(type(category_id) is int)

            # Read in image.
            im = Image.open(os.path.join(im_fp,'{}.jpg'.format(f)))
            (n_rows, n_cols, n_channels) = np.shape(im)

            # Ignore image if no bounding boxes
            if category_id == 0 and len(detect) == 0:
                empty_count += 1
                continue
            elif category_id == 0:
                empty_w_bbox_count += 1
                #print('ERROR: Image:{} Category:{} Length bbox:{}'.format(f, category_id, len(detect)))
            elif len(detect) == 0:
                nonempty_no_bbox_count += 1
                #print('ERROR: Image:{} Category:{} Length bbox:{}'.format(f, category_id, len(detect)))

            # If there are detections in image.
            for d in detect:
                category = d['category']
                [x, y, width, height] = d['bbox']
                bbox = (int(x*n_cols), int(y*n_rows), int((x+width)*n_cols), int(n_rows*(y+height)))
                conf = d['conf']

                self.id_lst.append(f)
                self.conf.append(conf)

                # Crop image with PIL.
                im_crop = im.crop(bbox)
                self.im_lst.append(self.transform(im_crop))

                # Check if category is human. (also 75)
                if category == '2' and category_id != HUMAN_CATEGORY_ID and category_id != 0:
                    #print('random human', category_id)
                    self.target_lst.append(HUMAN_CATEGORY_ID)
                # If animal or actual human or empty
                elif category == '1' or category_id == HUMAN_CATEGORY_ID or category_id == 0:
                    self.target_lst.append(category_id)
                else:
                    print('ERROR: Only categories 1/2:', category)

        print('Number of empty images with no bounding boxes:', empty_count)
        print('Number of empty images with bounding boxes:', empty_w_bbox_count)
        print('Number of nonempty images without bounding boxes:', nonempty_no_bbox_count)
        print('Final number of cropped images:', len(self.im_lst))
        assert(len(self.im_lst) == len(self.target_lst))
        assert(len(self.im_lst) == len(self.id_lst))
        assert(len(self.im_lst) == len(self.conf))


    def create_im_dict(self, annotations, detections):
        # Count?
        # 216 species but 276 in paper 571 max
        # Create dictionary with the key as image id and the values as
        # the bounding boxes and category id.
        # detections is a list of dictionaries {category 1/2, bbox, confidence}
        im_dict = {a['image_id']:{'category_id':a['category_id']} for a in annotations}

        # Iterate over bounding boxes info.
        for detect in detections:
            if im_dict.get(detect['id']) != None:
                if im_dict[detect['id']].get('detections') != None:
                    print('ERROR: More than one detection')
                im_dict[detect['id']]['detections'] = detect['detections']
        return im_dict


    def __len__(self):
        return len(self.im_lst)

    def __getitem__(self, idx):
        return {'image':self.im_lst[idx], 'target':self.target_lst[idx], 'id':self.id_lst[idx]}
