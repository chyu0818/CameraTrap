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
from data_reader import CameraTrapCropDataset
from triplet_loss import Classifier

def train(model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    criterion = nn.CrossEntropyLoss()
    model.train()# Set the model to training mode
    losses = []
    for batch_idx, data0 in enumerate(train_loader):
        data = data0['image']
        target = data0['target']
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        outputs = model(data)                # Make predictions
        loss = criterion(outputs, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step
        losses.append(loss.item())
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_dataset),
                100. * batch_idx / len(train_dataset), losses[-1]))
    return np.mean(losses)


def get_preds(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    predictions = []
    labels = []
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data0 in test_loader:
            data = data0['image']
            target = data0['target']
            data, target = data.to(device), target.to(device)
            output = model(data)
            predictions.extend(output)
            labels.extend(target)


cuda = torch.cuda.is_available()
BATCH_SIZE_TRAIN = 512
BATCH_SIZE_VAL = 512
LOG_INTERVAL = 20
NUM_CLASSES = 267
NUM_EPOCHS = 20
#np.random.seed(1)

normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
transform_val = T.Compose([T.Resize(size=(64,64)),
                       T.ToTensor(),
                       normalize])
transform_train = T.Compose([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                       T.RandomResizedCrop(size=(64,64),scale=(0.8, 1.0)),
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
weights_path = "models/10_64_512_triplet_aug/triplet_finetune_resnet_10.pt"

print('Train Data')
train_dataset = CameraTrapCropDataset(img_path, train_path, ann_path, bbox_path,
                                  percent_data, transform=transform_val)
print('\nVal Cis-Location Data')
val_cis_dataset = CameraTrapCropDataset(img_path, val_cis_path, ann_path, bbox_path,
                                  percent_data, transform=transform_val)

print('\nVal Trans-Location Data')
val_trans_dataset = CameraTrapCropDataset(img_path, val_trans_path, ann_path, bbox_path,
                                  percent_data, transform=transform_val)


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

model = Classifier(embedding_net)
model.load_state_dict(torch.load(weights_path))

for param in model.parameters():
    param.requires_grad = False


device = "cuda"
model.to(device)


