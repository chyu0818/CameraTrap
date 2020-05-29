import os
from torch.utils.data import Dataset
import random
import numpy as np
import torch
import json
from PIL import Image

HUMAN_CATEGORY_ID = 75
class CameraTrapCropTripletDataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples, will change later for location specific etc
    Test: Creates fixed triplets for testing
    """
    def __init__(self, im_fp, file_lst_fn, annotations_fn, bbox_fn, percent_data, train=True, transform=None):
        '''
        im_fp: filepath for images
        file_lst_fn: filename for list of images
        annotations_fn: filename for annotations
        bbox_fn: filename for detections
        '''
        self.im_fp = im_fp
        self.target_lst = []   # List of target categories.
        self.id_lst = []   # List of image ids corresponding to each bounding box.
        self.conf = []   # List of confidence values.
        self.transform = transform
        self.train = train

        # Load files.
        file_lst = np.load(file_lst_fn)['arr_0']
        file_lst = file_lst[:int(percent_data * len(file_lst))]
        print('Total number of files: ', len(file_lst))

        # Read in json files.
        with open(annotations_fn) as f:
            annotations0 = json.load(f)
        annotations = annotations0['annotations']
        all_categories = annotations0['categories']
        categories_dict = {all_categories[i]['id']:i for i in range(len(all_categories))}
        print('Number of classes:', len(categories_dict))
        with open(bbox_fn) as f1:
            detections = json.load(f1)['images']

        # Create dictionary by image id.
        im_dict = self.create_im_dict(annotations, detections)

        empty_count = 0
        empty_w_bbox_count = 0
        nonempty_no_bbox_count = 0

        no_detections_count = 0
        no_category_id_count = 0
        nonexistent_category_count = 0
        for f in file_lst:
            # Get detections and category id from dictionary.
            if im_dict[f].get('detections') == None:
                no_detections_count += 1
                continue
            detect = im_dict[f]['detections']
            if im_dict[f].get('category_id') == None:
                im_dict[f]['category_id'] = 0
                no_category_id_count += 1
            category_id = im_dict[f]['category_id']
            assert(type(category_id) is int)

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
                continue

            # If there are detections in image.
            for i in range(len(detect)):
                d = detect[i]
                category = d['category']
                conf = d['conf']

                # Check if category is human. (also 75)
                if category == '2' and category_id != HUMAN_CATEGORY_ID and category_id != 0:
                    #print('random human', category_id)
                    self.target_lst.append(categories_dict[HUMAN_CATEGORY_ID])
                    # ID: filename, index in detections, and bounding box coordinates
                    idd = '{}_{}_{}'.format(f, i, '-'.join([str(dd) for dd in d['bbox']]))
                    self.id_lst.append(idd)
                    self.conf.append(conf)
                # If animal or actual human or empty
                elif category == '1' or category_id == HUMAN_CATEGORY_ID or category_id == 0:
                    if categories_dict.get(category_id) != None:
                        self.target_lst.append(categories_dict[category_id])
                        idd = '{}_{}_{}'.format(f, i, '-'.join([str(dd) for dd in d['bbox']]))
                        self.id_lst.append(idd)
                        self.conf.append(conf)
                    else:
                        nonexistent_category_count += 1
                else:
                    print('ERROR: Only categories 1/2:', category)

        print('\nNumber of empty images with no bounding boxes:', empty_count)
        print('Number of empty images with bounding boxes:', empty_w_bbox_count)
        print('Number of nonempty images without bounding boxes:', nonempty_no_bbox_count)

        print('\nNumber of images without detections (even []) listed:', no_detections_count)
        print('Number of images without categories:', no_category_id_count)
        print('Number of categories that do not exist:', nonexistent_category_count)

        print('\nFinal number of cropped images:', len(self.id_lst), '\n')
        assert(len(self.id_lst) == len(self.target_lst))
        assert(len(self.id_lst) == len(self.conf))

        self.targets_set = set(np.asarray(self.target_lst))
        self.target_to_indices = {target: np.where(np.asarray(self.target_lst) == np.asarray(target))[0]
                                     for target in self.targets_set}
        # Ignore classes with only 1 image. 
        inds_remove = []
        for tar in self.target_to_indices:
            if len(self.target_to_indices[tar]) == 1:
                inds_remove.append(self.target_to_indices[tar][0])
                self.target_to_indices[tar] = []

        # Sort indices backwards.
        inds_remove.sort(reverse=True)
        print('Number of removed images:', len(inds_remove))
        print(inds_remove)
        for ind in inds_remove:
            self.target_lst.pop(ind)
            self.id_lst.pop(ind)
            self.conf.pop(ind)

        self.targets_set = set(np.asarray(self.target_lst))
        self.target_to_indices = {target: np.where(np.asarray(self.target_lst) == np.asarray(target))[0]
                                     for target in self.targets_set}

        # For testing, use fixed triplets. 
        if not self.train:
            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.target_to_indices[self.target_lst[i]]),
                         random_state.choice(self.target_to_indices[
                                                 np.random.choice(
                                                     list(self.targets_set - set([self.target_lst[i]]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.id_lst))]
            self.test_triplets = triplets


    def create_im_dict(self, annotations, detections):
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
        return len(self.id_lst)

    def __getitem__(self, idx):
        if self.train:
            # Anchor
            id1, target1 = self.id_lst[idx], self.target_lst[idx]

            # Positive
            positive_index = idx
            while positive_index == idx:
                positive_index = np.random.choice(self.target_to_indices[target1])

            # Negative
            negative_label = np.random.choice(list(self.targets_set - set([target1])))
            negative_index = np.random.choice(self.target_to_indices[negative_label])

            id2 = self.id_lst[positive_index]
            id3 = self.id_lst[negative_index]
        else:
            id1 = self.id_lst[self.test_triplets[idx][0]]
            id2 = self.id_lst[self.test_triplets[idx][1]]
            id3 = self.id_lst[self.test_triplets[idx][2]]

        im1 = self.transform(Image.open(os.path.join(self.im_fp,'{}.jpg'.format(id1))))
        im2 = self.transform(Image.open(os.path.join(self.im_fp,'{}.jpg'.format(id2))))
        im3 = self.transform(Image.open(os.path.join(self.im_fp,'{}.jpg'.format(id3))))
        return (im1, im2, im3), []


class CameraTrapCropDataset(Dataset):
    def __init__(self, im_fp, file_lst_fn, annotations_fn, bbox_fn, percent_data, transform=None):
        '''
        im_fp: filepath for images
        file_lst_fn: filename for list of images
        annotations_fn: filename for annotations
        bbox_fn: filename for detections
        '''
        self.im_fp = im_fp
        self.target_lst = []   # List of target categories.
        self.id_lst = []   # List of image ids corresponding to each bounding box.
        self.conf = []   # List of confidence values.
        self.transform = transform

        # Randomize files.
        file_lst = np.load(file_lst_fn)['arr_0']
        np.random.shuffle(file_lst)
        file_lst = file_lst[:int(percent_data * len(file_lst))]
        print('Total number of files: ', len(file_lst))

        # Read in json files.
        with open(annotations_fn) as f:
            annotations0 = json.load(f)
        annotations = annotations0['annotations']
        all_categories = annotations0['categories']
        categories_dict = {all_categories[i]['id']:i for i in range(len(all_categories))}
        print('Number of classes:', len(categories_dict))
        with open(bbox_fn) as f1:
            detections = json.load(f1)['images']

        # Create dictionary by image id.
        im_dict = self.create_im_dict(annotations, detections)

        empty_count = 0
        empty_w_bbox_count = 0
        nonempty_no_bbox_count = 0

        no_detections_count = 0
        no_category_id_count = 0
        nonexistent_category_count = 0
        for f in file_lst:
            # Get detections and category id from dictionary.
            if im_dict[f].get('detections') == None:
                no_detections_count += 1
                continue
            detect = im_dict[f]['detections']
            if im_dict[f].get('category_id') == None:
                im_dict[f]['category_id'] = 0
                no_category_id_count += 1
            category_id = im_dict[f]['category_id']
            assert(type(category_id) is int)

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
                continue

            # If there are detections in image.
            for i in range(len(detect)):
                d = detect[i]
                category = d['category']
                conf = d['conf']

                # Check if category is human. (also 75)
                if category == '2' and category_id != HUMAN_CATEGORY_ID and category_id != 0:
                    #print('random human', category_id)
                    self.target_lst.append(categories_dict[HUMAN_CATEGORY_ID])
                    # ID: filename, index in detections, and bounding box coordinates
                    idd = '{}_{}_{}'.format(f, i, '-'.join([str(dd) for dd in d['bbox']]))
                    self.id_lst.append(idd)
                    self.conf.append(conf)
                # If animal or actual human or empty
                elif category == '1' or category_id == HUMAN_CATEGORY_ID or category_id == 0:
                    if categories_dict.get(category_id) != None:
                        self.target_lst.append(categories_dict[category_id])
                        idd = '{}_{}_{}'.format(f, i, '-'.join([str(dd) for dd in d['bbox']]))
                        self.id_lst.append(idd)
                        self.conf.append(conf)
                    else:
                        nonexistent_category_count += 1
                else:
                    print('ERROR: Only categories 1/2:', category)

        print('\nNumber of empty images with no bounding boxes:', empty_count)
        print('Number of empty images with bounding boxes:', empty_w_bbox_count)
        print('Number of nonempty images without bounding boxes:', nonempty_no_bbox_count)

        print('\nNumber of images without detections (even []) listed:', no_detections_count)
        print('Number of images without categories:', no_category_id_count)
        print('Number of categories that do not exist:', nonexistent_category_count)

        print('\nFinal number of cropped images:', len(self.id_lst), '\n')
        assert(len(self.id_lst) == len(self.target_lst))
        assert(len(self.id_lst) == len(self.conf))


    def create_im_dict(self, annotations, detections):
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
        return len(self.id_lst)

    def __getitem__(self, idx):
        # Load image.
        im_crop = Image.open(os.path.join(self.im_fp,'{}.jpg'.format(self.id_lst[idx])))
        return {'image':self.transform(im_crop), 'target':self.target_lst[idx], 'id':self.id_lst[idx]}


class CameraTrapDataset(Dataset):
    def __init__(self, im_fp, file_lst_fn, annotations_fn, bbox_fn, percent_data, transform=None):
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

        # Randomize files.
        file_lst = np.load(file_lst_fn)['arr_0']
        np.random.shuffle(file_lst)
        file_lst = file_lst[:int(percent_data * len(file_lst))]
        print('Total number of files: ', len(file_lst))

        # Read in json files.
        with open(annotations_fn) as f:
            annotations0 = json.load(f)
        annotations = annotations0['annotations']
        all_categories = annotations0['categories']
        categories_dict = {all_categories[i]['id']:i for i in range(len(all_categories))}
        print('Number of classes:', len(categories_dict))
        with open(bbox_fn) as f1:
            detections = json.load(f1)['images']

        # Create dictionary by image id.
        im_dict = self.create_im_dict(annotations, detections)

        empty_count = 0
        empty_w_bbox_count = 0
        nonempty_no_bbox_count = 0

        no_detections_count = 0
        no_category_id_count = 0
        nonexistent_category_count = 0
        for f in file_lst:
            # Get detections and category id from dictionary.
            if im_dict[f].get('detections') == None:
                no_detections_count += 1
                continue
            detect = im_dict[f]['detections']
            if im_dict[f].get('category_id') == None:
                im_dict[f]['category_id'] = 0
                no_category_id_count += 1
            category_id = im_dict[f]['category_id']
            assert(type(category_id) is int)

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
                continue

            # Read in image.
            im = Image.open(os.path.join(im_fp,'{}.jpg'.format(f)))
            (n_rows, n_cols, n_channels) = np.shape(im)

            # If there are detections in image.
            for i in range(len(detect)):
                d = detect[i]
                category = d['category']
                [x, y, width, height] = d['bbox']
                bbox = (int(x*n_cols), int(y*n_rows), int((x+width)*n_cols), int(n_rows*(y+height)))
                conf = d['conf']


                # Check if category is human. (also 75)
                if category == '2' and category_id != HUMAN_CATEGORY_ID and category_id != 0:
                    #print('random human', category_id)
                    self.target_lst.append(categories_dict[HUMAN_CATEGORY_ID])

                    # Crop image with PIL.
                    im_crop = im.crop(bbox)
                    self.im_lst.append(self.transform(im_crop))
                    # ID: filename, index in detections, and bounding box coordinates
                    self.id_lst.append('{}_{}_{}'.format(f, i, '-'.join([str(dd) for dd in d['bbox']])))
                    self.conf.append(conf)
                # If animal or actual human or empty
                elif category == '1' or category_id == HUMAN_CATEGORY_ID or category_id == 0:
                    if categories_dict.get(category_id) != None:
                        self.target_lst.append(categories_dict[category_id])
                        # Crop image with PIL.
                        im_crop = im.crop(bbox)
                        self.im_lst.append(self.transform(im_crop))
                        self.id_lst.append('{}_{}_{}'.format(f, i, '-'.join([str(dd) for dd in d['bbox']])))
                        self.conf.append(conf)
                    else:
                        nonexistent_category_count += 1
                else:
                    print('ERROR: Only categories 1/2:', category)

        print('\nNumber of empty images with no bounding boxes:', empty_count)
        print('Number of empty images with bounding boxes:', empty_w_bbox_count)
        print('Number of nonempty images without bounding boxes:', nonempty_no_bbox_count)

        print('\nNumber of images without detections (even []) listed:', no_detections_count)
        print('Number of images without categories:', no_category_id_count)
        print('Number of categories that do not exist:', nonexistent_category_count)

        print('\nFinal number of cropped images:', len(self.im_lst), '\n')
        assert(len(self.im_lst) == len(self.target_lst))
        assert(len(self.im_lst) == len(self.id_lst))
        assert(len(self.im_lst) == len(self.conf))


    def create_im_dict(self, annotations, detections):
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
