from __future__ import print_function
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms as T
from data_reader1 import CameraTrapDataset
import json
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import numpy as np

def train(model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()# Set the model to training mode
    losses = []
    log_interval = 100
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)                # Make predictions
        loss = F.nll_loss(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step
        losses.append(loss.item())
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader.sampler), losses[-1]))
    return np.mean(losses)


def test(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(target)

    test_loss /= total

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total,
        100. * correct / total))
    return test_loss

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
model = models.resnet18(pretrained=True)

# Fix everything but final layer
for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Linear(2048, 572)
model.to(device)

normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
#transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
# Not sure if Crop is required
transform = T.Compose([T.Resize((256,256)), T.ToTensor(), normalize])

train_path = 'X_train.npz'
val_path = 'X_val.npz'
img_path = '../efs/train'
ann_path = '../efs/iwildcam2020_train_annotations.json'
bbox_path = '../efs/iwildcam2020_megadetector_results.json'

with open(ann_path) as f:
    ann = json.load(f)

with open(bbox_path) as f:
    bbox = json.load(f)

train_dataset = CameraTrapDataset(img_path, train_path, ann_path, bbox_path,
                                  transform=transform)
val_dataset = CameraTrapDataset(img_path, val_path, ann_path, bbox_path,
                                  transform=transform)

train_loader = torch.utils.data.DataLoader(
        train_dataset[:1], batch_size=1, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
        train_dataset[:1], batch_size=1, shuffle=True
)

lr = 1
optimizer = optim.Adadelta(model.parameters(), lr=lr)

step = 1
gamma = 0.7
scheduler = StepLR(optimizer, step_size=step, gamma=gamma)

# Training loop
train_losses = []
test_losses = []
epochs = 1
for epoch in range(1, epochs + 1):
    train_loss = train(model, device, train_loader, optimizer, epoch)
    test_loss = test(model, device, val_loader)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    scheduler.step()    # learning rate scheduler

# You may optionally save your model at each epoch here
np.save("train_loss.npy", np.array(train_losses))
np.save("test_loss.npy", np.array(test_losses))

print("Final Performance!")
print("Validation Set:")
test(model, device, val_loader)
print("Training Set:")
test(model, device, train_loader)
torch.save(model.state_dict(), "baseline.pt")
