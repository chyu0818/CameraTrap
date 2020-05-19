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
    train_locs = locs[:round(0.5*n)]
    val_locs = locs[round(0.5*n):round(0.55*n)]
    test_locs = locs[round(0.55*n):]

    # Get list of images for each set first and then we'll split annotations
    train_images = set([])
    sequences = set([])
    val_images = set([])
    test_images = set([])
    test_cis_images = set([])
    map_to_seq = {}

    for item in images:
        if item['location'] in train_locs:
            # even days get assigned to test set
            if int(item['datetime'][8:10]) % 3 == 1 or int(item['datetime'][8:10]) % 3 == 0:
                train_images.add(item['id'])
                map_to_seq[item['id']] = item['seq_id']
                if item['seq_id'] not in sequences:
                    sequences.add(item['seq_id'])
            else:
                test_cis_images.add(item['id'])
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

    test_cis_count = 0
    test_trans_count = 0

    for item in annotations:
        if item['image_id'] in train_images:
            train_annotations.append(item['image_id'])
            y_train.append(item['category_id'])
        elif item['image_id'] in val_images:
            val_annotations.append(item['image_id'])
            y_val.append(item['category_id'])
        else:
            if item['image_id'] in test_cis_images:
                test_cis_count += 1
            else:
                test_trans_count += 1
            test_annotations.append(item['image_id'])
            y_test.append(item['category_id'])

    cis_val_count = 0
    sequences = list(np.random.permutation(list(sequences)))
    new_val_annotations = []
    new_val_labels = []
    while cis_val_count < 0.05*len(train_annotations):
        curr = sequences.pop()
        for i, item in enumerate(train_annotations):
            if map_to_seq[item] == curr:
                new_val_annotations.append(item)
                new_val_labels.append(y_train[i])
                cis_val_count += 1

    val_annotations.extend(new_val_annotations)
    y_val.extend(new_val_labels)

    X_train = []
    new_y_train = []
    new_val_annotations = set(new_val_annotations)
    for i, item in enumerate(train_annotations):
        if item not in new_val_annotations:
            X_train.append(item)
            new_y_train.append(y_train[i])

    print("training", len(X_train))
    print("trans_val", len(val_annotations))
    print("cis_val", cis_val_count)
    print("trans_test", test_trans_count)
    print("cis_test", test_cis_count)

    # Move 5% to validation
    # X_train, X_val, y_train, label_val = train_test_split(train_annotations, y_train, test_size = 0.05)
    # val_annotations.extend(X_val)
    # y_val.extend(label_val)
    assert (len(X_train) == len(new_y_train))
    assert (len(val_annotations) == len(y_val))
    assert (len(test_annotations) == len(y_test))
    assert (len(X_train) != 0)
    assert (len(y_val) != 0)
    assert (len(y_test) != 0)
    return X_train, val_annotations, test_annotations, new_y_train, y_val, y_test

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
