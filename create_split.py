# We will randomly split the 552 locations into a test, val and train
# Further train set is then split into a test, val, and train
import numpy as np
import json
from sklearn.model_selection import train_test_split

def create_split(annotations, images):
    locs = []
    for item in images:
        if item['location'] not in locs:
            locs.append(item['location'])
    locs = np.random.permutation(locs)

    # Split locations into train/val/test 70/20/10
    n = len(locs)
    train_locs = locs[:round(0.7*n)]
    val_locs = locs[round(0.7*n):round(0.9*n)]
    test_locs = locs[round(0.9*n):]

    # Get list of images for each set first and then we'll split annotations
    train_images = set([])
    val_images = set([])
    test_images = set([])

    for item in images:
        if item['location'] in train_locs:
            # odd days get assigned to test set
            if int(item['datetime'][8:10]) % 2 == 0:
                train_images.add(item['id'])
            else:
                test_images.add(item['id'])
        elif item['location'] in val_locs:
            val_images.add(item['id'])
        else:
            test_images.add(item['id'])
    print('Images Split Done!')
    train_annotations = []
    y_train = []
    val_annotations = []
    y_val = []
    test_annotations = []
    y_test = []

    print('Splitting Annotations..')

    for item in annotations:
        if item['image_id'] in train_images:
            train_annotations.append(item['id'])
            y_train.append(item['category_id'])
        elif item['image_id'] in val_images:
            val_annotations.append(item['id'])
            y_val.append(item['category_id'])
        else:
            test_annotations.append(item['id'])
            y_test.append(item['category_id'])

    print(len(train_annotations))
    print(len(test_annotations))
    print(len(val_annotations))

    # Move 5% to validation
    X_train, X_val, y_train, label_val = train_test_split(train_annotations, y_train, test_size = 0.05)
    val_annotations.extend(X_val)
    y_val.extend(label_val)
    return X_train, val_annotations, test_annotations, y_train, y_val, y_test

if __name__ == '__main__':
    with open('iwildcam2020_train_annotations.json') as f:
        data = json.load(f)
    train, val, test, y_train, y_val, y_test = create_split(data['annotations'], data['images'])
    np.savez("X_train.npz", train)
    np.savez("X_val.npz", val)
    np.savez("X_test.npz", test)
    np.savez("y_train.npz", y_train)
    np.savez("y_val.npz", y_val)
    np.savez("y_test.npz", y_test)
