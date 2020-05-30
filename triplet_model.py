import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms as T
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
import json
import torch.nn.functional as F
import time
from trainer import fit
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from data_reader import CameraTrapCropTripletDataset
from triplet_loss import TripletNet, TripletLoss

cuda = torch.cuda.is_available()
BATCH_SIZE_TRAIN = 512
BATCH_SIZE_VAL = 512
LOG_INTERVAL = 20
NUM_CLASSES = 267
NUM_EPOCHS = 20
np.random.seed(1)

normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
transform_val = T.Compose([T.Resize(size=(256,256)),
                       T.ToTensor(),
                       normalize])
transform_train = T.Compose([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                       T.RandomResizedCrop(size=(256,256),scale=(0.8, 1.0)),
                       T.ToTensor(),
                       normalize])

train_path = 'X_train.npz'
val_cis_path = 'X_val_cis.npz'
val_trans_path = 'X_val_trans.npz'
img_path = '../efs/train_crop'
ann_path = '../efs/iwildcam2020_train_annotations.json'
bbox_path = '../efs/iwildcam2020_megadetector_results.json'
percent_data = 1
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {} 

print('Train Data')
train_dataset = CameraTrapCropTripletDataset(img_path, train_path, ann_path, bbox_path,
                                  percent_data, conf_threshold=0.5, transform=transform_train, train=True)
print('\nVal Cis-Location Data')
val_cis_dataset = CameraTrapCropTripletDataset(img_path, val_cis_path, ann_path, bbox_path,
                                  percent_data, conf_threshold=0.5, transform=transform_val, train=False)

print('\nVal Trans-Location Data')
val_trans_dataset = CameraTrapCropTripletDataset(img_path, val_trans_path, ann_path, bbox_path,
                                  percent_data, conf_threshold=0.5, transform=transform_val, train=False)


train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, **kwargs
)
val_cis_loader = torch.utils.data.DataLoader(
        val_cis_dataset, batch_size=BATCH_SIZE_VAL, shuffle=True, **kwargs
)
val_trans_loader = torch.utils.data.DataLoader(
        val_trans_dataset, batch_size=BATCH_SIZE_VAL, shuffle=True, **kwargs
)

embedding_net = models.resnet18(pretrained=True)
# for param in embedding_net.parameters():
#     param.requires_grad = False
embedding_net.fc = torch.nn.Linear(512, 1000)

model = TripletNet(embedding_net)
if cuda:
    model.cuda()

loss_fn = TripletLoss()
lr = 1
optimizer = optim.Adadelta(model.parameters(), lr=lr)
step = 1
gamma = 0.7
scheduler = lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)

fit(train_loader, val_cis_loader, val_trans_loader, model, loss_fn, optimizer, scheduler, NUM_EPOCHS, cuda, LOG_INTERVAL)


