from __future__ import print_function
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms as T
from data_reader import CameraTrapDataset
import json
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt

BATCH_SIZE_TRAIN = 1000
BATCH_SIZE_VAL = 1000
LOG_INTERVAL = 10
NUM_CLASSES = 267 # 267??
NUM_EPOCHS = 20

def test(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    error = np.zeros((267, 1))
    total = np.zeros((267, 1))
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data0 in test_loader:
            data = data0['image']
            target = data0['target']
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            for i in range(len(pred)):
                if pred[i] != target[i]:
                    error[target[i]] += 1
                total[target[i]] += 1
    return error, total

#use_cuda = True
#device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}
model = models.resnet18(pretrained=True)

# Fix everything but final layer
for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Linear(512, NUM_CLASSES)
model.to(device)

normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
#transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
# Not sure if Crop is required
transform = T.Compose([T.Resize((128,128)), T.ToTensor(), normalize])

train_path = 'X_train.npz'
val_path = 'X_val.npz'
img_path = '../efs/train'
ann_path = '../efs/iwildcam2020_train_annotations.json'
bbox_path = '../efs/iwildcam2020_megadetector_results.json'
model_path = "baseline1.pt"
percent_data = 1
# ~70k train, ~20k val

print('Train Data')
train_dataset = CameraTrapDataset(img_path, train_path, ann_path, bbox_path,
                                  percent_data, transform=transform)
print('\nVal Data')
val_dataset = CameraTrapDataset(img_path, val_path, ann_path, bbox_path,
                                  percent_data, transform=transform)

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, **kwargs
)
val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE_VAL, shuffle=True, **kwargs
)

model.load_state_dict(torch.load(model_path))
train_err, train_total = test(model, device, train_loader)
val_err, val_total = test(model, device, val_loader)

log_train_err = []
log_train_counts = []
for i in range(len(train_total)):
    if train_total[i] != 0:
        log_train_err.append(np.log(train_err[i] / train_total[i]))
        log_train_counts.append(np.log(train_total[i]))

log_val_err = []
log_val_counts = []
for i in range(len(train_total)):
    if train_total[i] != 0 and val_total[i] != 0:
        log_val_err.append(np.log(val_err[i] / val_total[i]))
        log_train_counts.append(np.log(val_total[i]))

plt.scatter(log_train_counts, log_train_err, marker="o")
plt.scatter(log_val_counts, log_val_err, marker="v")
plt.xlabel("Log Scale Number of Training Examples For the Class")
plt.ylabel("Log Scale Error Rate")
plt.legend(["Train", "Validation"])
plt.show()
