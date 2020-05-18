import os
import numpy as np
import json
annotations_fn = 'iwildcam2020_train_annotations.json'
# Read in json files.
with open(annotations_fn) as f:
    annotations0 = json.load(f)
all_categories = annotations0['categories']
categories_to_new = {all_categories[i]['id']:i for i in range(len(all_categories))}

new_cat_train = [32, 1, 36, 0, 189, 188, 48, 1, 36, 188, 1, 32, 45, 1, 48, 32, 65, 1]
new_cat_val = [17, 1, 187, 147, 31, 44, 195, 2, 61, 8, 187, 45, 195, 188, 44, 0, 4, 44]

old_cat_train = {idx:all_categories[idx]['name'] for idx in new_cat_train}
print(old_cat_train)
old_cat_val = {idx:all_categories[idx]['name'] for idx in new_cat_val}
print(old_cat_val)
