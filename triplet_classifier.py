import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms as T
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import json
import torch.nn.functional as F
import time
from trainer import fit
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from data_reader import CameraTrapCropDataset, CameraTrapEmbeddingDataset
from triplet_loss import Embedder, Classifier

def train(model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    criterion = nn.CrossEntropyLoss()
    model.train()# Set the model to training mode
    losses = []
    for batch_idx, data0 in enumerate(train_loader):
        data = data0['embedding']
        target = data0['target']
        data, target = data.to(device), target.to(device)
        target = target.long()
        optimizer.zero_grad()               # Clear the gradient
        outputs = model(data.float())                # Make predictions
        loss = criterion(outputs, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step
        losses.append(loss.item())
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_dataset),
                100. * batch_idx / len(train_dataset), losses[-1]))
    return np.mean(losses)

def test(model, device, test_loader):
    criterion = nn.CrossEntropyLoss(reduction='sum')
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data0 in test_loader:
            data = data0['embedding']
            target = data0['target']
            data, target = data.to(device), target.to(device)
            target = target.long()
            output = model(data.float())
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(target)

    test_loss /= total

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total,
        100. * correct / total))
    return test_loss

def get_embeddings(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    embeddings = np.zeros((len(test_loader.dataset), 1000))
    labels = np.zeros(len(test_loader.dataset))
    k = 0
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data0 in test_loader:
            data = data0['image']
            target = data0['target']
            data, target = data.to(device), target.to(device)
            embeddings[k:k + len(data)] = model(data).cpu().numpy()
            labels[k: k + len(data)] = target.cpu().numpy()
            k += len(data)
    return embeddings, labels


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

model = Embedder(embedding_net)
model.load_state_dict(torch.load(weights_path))

for param in model.parameters():
    param.requires_grad = False

device = "cuda"
model.to(device)

print("Obtaining Embeddings!")

X_train, y_train = get_embeddings(model, device, train_loader)
print("Got Train Embeddings!", X_train.shape, y_train.shape)

X_val_cis, y_val_cis = get_embeddings(model, device, val_cis_loader)
print("Got Val Cis Embeddings")

X_val_trans, y_val_trans = get_embeddings(model, device, val_trans_loader)
print("Got Val Trans Embeddings")

train_dataset = CameraTrapEmbeddingDataset(X_train, y_train)
val_cis_dataset = CameraTrapEmbeddingDataset(X_val_cis, y_val_cis)
val_trans_dataset = CameraTrapEmbeddingDataset(X_val_trans, y_val_trans)

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, **kwargs
)
val_cis_loader = torch.utils.data.DataLoader(
        val_cis_dataset, batch_size=BATCH_SIZE_VAL, shuffle=True, **kwargs
)
val_trans_loader = torch.utils.data.DataLoader(
        val_trans_dataset, batch_size=BATCH_SIZE_VAL, shuffle=True, **kwargs
)

model = Classifier()
model.to(device)


print("Starting Training!")

lr = 1
optimizer = optim.Adadelta(model.parameters(), lr=lr)

step = 1
gamma = 0.7
scheduler = StepLR(optimizer, step_size=step, gamma=gamma)

# Training loop
train_losses = []
test_cis_losses = []
test_trans_losses = []
start = time.time()
for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = train(model, device, train_loader, optimizer, epoch)
    test_cis_loss = test(model, device, val_cis_loader)
    test_trans_loss = test(model, device, val_trans_loader)
    train_losses.append(train_loss)
    test_cis_losses.append(test_cis_loss)
    test_trans_losses.append(test_trans_loss)
    scheduler.step()    # learning rate scheduler
    torch.save(model.state_dict(), "triplet_classifier_{}_{}.pt".format(percent_data,epoch))
print('Train Time:', time.time()-start)
# You may optionally save your model at each epoch here
np.save("train_triplet_classifier_loss{}.npy".format(percent_data), np.array(train_losses))
np.save("test_triplet_classifier_cis_loss{}.npy".format(percent_data), np.array(test_cis_losses))
np.save("test_triplet_classifier_trans_loss{}.npy".format(percent_data), np.array(test_trans_losses))

print("\nFinal Performance!")
print("Validation Set (cis):")
test(model, device, val_cis_loader)
print("Validation Set (trans):")
test(model, device, val_trans_loader)
print("Training Set:")
test(model, device, train_loader)






