import numpy as np
import json

with open('iwildcam2020_train_annotations.json') as f:
    data = json.load(f)

train_data = set(np.load("X_train.npz")["arr_0"])
print("Train Data Loaded!")
cis_locs = set([])
info = data['images']
print("Finding Cis Locations.")
for item in info:
    if item['id'] in train_data:
        if item['location'] not in cis_locs:
            cis_locs.add(item['location'])
print('Found All Cis Locations')
print(len(cis_locs))

validation_data = set(np.load("X_val.npz")["arr_0"])
val_cis = []
val_trans = []

test_data = set(np.load("X_test.npz")["arr_0"])
test_cis = []
test_trans = []
print('Splitting Test and Validation Sets')
for item in info:
    if item['id'] in validation_data:
        if item['location'] in cis_locs:
            val_cis.append(item['id'])
        else:
            val_trans.append(item['id'])
    elif item['id'] in test_data:
        if item['location'] in cis_locs:
            test_cis.append(item['id'])
        else:
            test_trans.append(item['id'])

np.savez("X_val_cis.npz", val_cis)
np.savez("X_test_cis.npz", test_cis)
np.savez("X_val_trans.npz", val_trans)
np.savez("X_test_trans.npz", test_trans)

y_val_cis = []
y_val_trans = []
y_test_cis = []
y_test_trans = []

test_data = np.load("X_test.npz")["arr_0"]
val_data = np.load("X_val.npz")["arr_0"]

test_labels = np.load("y_test.npz")["arr_0"]
val_labels = np.load("y_val.npz")["arr_0"]

val_cis = set(val_cis)
test_cis = set(test_cis)

for i, item in enumerate(test_data):
    if item in test_cis:
        y_test_cis.append(test_labels[i])
    else:
        y_test_trans.append(test_labels[i])

np.savez("y_test_cis.npz", y_test_cis)
np.savez("y_test_trans.npz", y_test_trans)

for i, item in enumerate(val_data):
    if item in val_cis:
        y_val_cis.append(val_labels[i])
    else:
        y_val_trans.append(val_labels[i])

np.savez("y_val_cis.npz", y_val_cis)
np.savez("y_val_trans.npz", y_val_trans)
