# We will randomly split the 552 locations into a test, val and train
# Further train set is then split into a test, val, and train
import numpy as np
import json

def create_split(annotations, images):
    locs = np.random.permutation([0 for i in range(552)])

    # Split locations into train/val/test 70/20/10
    train_locs = locs[:round(0.7*552)]
    val_locs = locs[round(0.7*552):round(0.9*552)]
    test_locs = locs[round(0.9*552):]

    # Get list of images for each set first and then we'll split annotations
    train_images = []
    val_images = []
    test_images = []

    for item in images:
        if item['location'] in train_locs:
            train_images.append(item['id'])
        elif item['location'] in val_locs:
            val_images.append(item['id'])
        else:
            test_images.append(item['id'])

    train_annotations = []
    val_annotations = []
    test_annotations = []

    for item in annotations:
        if item['image_id'] in train_images:
            train_annotations.append(item['id'])
        elif item['image_id'] in val_images:
            val_annotations.append(item['id'])
        else:
            test_annotations.append(item['id'])

    # Now we need to randomly select 20% to move to validation and 10% to move
    # to test
    n = len(train_annotations)
    train_annotations = np.random.permutation(train_annotations)
    val_annotations = val_annotations.extend(train_annotations[:round(0.2*n)])
    test_annotations = test_annotations.extend(train_annotations[round(0.2*n):round(0.3*n)])
    train_annotations = train_annotations[round(0.3*n):]
    return train_annotations, val_annotations, test_annotations

if __name__ == '__main__':
    with open('iwildcam2020_train_annotations.json') as f:
        data = json.load(f)
    train, val, test = create_split(data['annotations'], data['images'])
    np.save("train.npy", train)
    np.save("validation.npy", val)
    np.save("test.npy", test)
