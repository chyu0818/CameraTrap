from __future__ import print_function
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms as T
from data_reader import CameraTrapCropDataset
import json
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
import os

BATCH_SIZE_TRAIN = 1000
BATCH_SIZE_VAL = 1000
LOG_INTERVAL = 10
NUM_CLASSES = 267 # 267??
NUM_EPOCHS = 20

def test(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    error = np.zeros(267)
    total = np.zeros(267)
    print(error.shape)
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data0 in test_loader:
            data = data0['image']
            target = data0['target']
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(target)
            for i in range(len(pred)):
                if pred[i,0] != target[i]:
                    error[target[i]] += 1
                total[target[i]] += 1
    test_loss /= total
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total,
        100. * correct / total))
    return error, total, test_loss

# Plots 9 examples from the test set where the classifier made a mistake.
def plot_mistakes(model, device, test_loader, save_fn):
    annotations_fn = 'iwildcam2020_train_annotations.json'
    with open(annotations_fn) as f:
        annotations0 = json.load(f)
    all_categories = annotations0['categories']
    model.eval()    # Set the model to inference mode
    img_path = '../efs/train'
    lim_mistakes = 9
    mistakes = []
    fig, axes = plt.subplots(3, 3)
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data0 in test_loader:
            data = data0['image']
            target = data0['target']
            idd = data0['id']
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            for i in range(len(pred)):
                if pred[i,0] != target[i]:
                    mistakes.append(idd[i])
                    # Split up ID to find filename and bounding box.
                    idd_lst = idd[i].split('_')
                    fn = idd_lst[0]
                    # Load image.
                    im = Image.open(os.path.join(img_path,'{}.jpg'.format(fn)))
                    (n_rows, n_cols, n_channels) = np.shape(im)
                    # Find bounding box.
                    [x, y, width, height] = [float(dim) for dim in idd_lst[2].split('-')]
                    bbox = (int(x*n_cols), int(y*n_rows), int((x+width)*n_cols), int(n_rows*(y+height)))
                    # Crop.
                    im_crop = im.crop(bbox)
                    # Plot.
                    ax = axes[(len(mistakes)-1)//3, (len(mistakes)-1)%3]
                    ax.imshow(im_crop, cmap='gray')
                    ax.set_title('Actual: {} Pred: {}'.format(all_categories[idx]['name'][target[i]], all_categories[idx]['name'][pred[i,0]]))
                    if len(mistakes) >= lim_mistakes:
                        plt.tight_layout()
                        plt.savefig(save_fn)
                        return mistakes
    return

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}
model = models.resnet18(pretrained=True)

model.fc = torch.nn.Linear(512, NUM_CLASSES)
model.to(device)

normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
#transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
# Not sure if Crop is required
transform = T.Compose([T.Resize((64,64)), T.ToTensor(), normalize])

train_path = 'X_train.npz'
val_cis_path = 'X_cis_val.npz'
val_trans_path = 'X_trans_val.npz'
img_path = '../efs/train_crop'
ann_path = '../efs/iwildcam2020_train_annotations.json'
bbox_path = '../efs/iwildcam2020_megadetector_results.json'
model_path = "baseline1_4.pt"
percent_data = 1
# ~70k train, ~20k val

print('Train Data')
train_dataset = CameraTrapCropDataset(img_path, train_path, ann_path, bbox_path,
                                  percent_data, transform=transform)
print('\nVal Data (cis)')
val_cis_dataset = CameraTrapCropDataset(img_path, val_cis_path, ann_path, bbox_path,
                                  percent_data, transform=transform)

print('\nVal Data (trans)')
val_trans_dataset = CameraTrapCropDataset(img_path, val_trans_path, ann_path, bbox_path,
                                  percent_data, transform=transform)

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, **kwargs
)
val_cis_loader = torch.utils.data.DataLoader(
        val_cis_dataset, batch_size=BATCH_SIZE_VAL, shuffle=True, **kwargs
)
val_trans_loader = torch.utils.data.DataLoader(
        val_trans_dataset, batch_size=BATCH_SIZE_VAL, shuffle=True, **kwargs
)

num_val_cis = 3307
num_val_trans = 6382
num_val_cis_frac = num_val_cis / (num_val_cis + num_val_trans)
num_val_trans_frac = num_val_trans/ (num_val_cis + num_val_trans)

model.load_state_dict(torch.load(model_path))
print('Training set:')
train_err, train_total, train_loss = test(model, device, train_loader)
print('Validation set (cis):')
val_cis_err, val_cis_total, val_cis_loss = test(model, device, val_cis_loader)
print('Validation set (trans):')
val_trans_err, val_trans_total, val_trans_loss = test(model, device, val_trans_loader)
val_err = val_cis_err + val_trans_err
val_total = val_cis_total + val_trans_total
val_loss = num_val_cis_frac * val_cis_loss + num_val_trans_frac * val_trans_loss
print('Overall validation set loss:', val_loss)

log_train_err = []
log_train_counts = []
for i in range(len(train_total)):
    if train_total[i] != 0:
        log_train_err.append(train_err[i] / train_total[i])
        log_train_counts.append(train_total[i])
log_val_cis_err = []
log_val_cis_counts = []
log_val_trans_err = []
log_val_trans_counts = []
log_val_err = []
log_val_counts = []
for i in range(len(train_total)):
    if train_total[i] != 0 and val_cis_total[i] != 0:
        if val_cis_err[i] != 0:
            log_val_cis_err.append(val_cis_err[i] / val_cis_total[i])
        else:
            log_val_cis_err.append(0)
        log_val_cis_counts.append(train_total[i])
    if train_total[i] != 0 and val_trans_total[i] != 0:
        if val_trans_err[i] != 0:
            log_val_trans_err.append(val_trans_err[i] / val_trans_total[i])
        else:
            log_val_trans_err.append(0)
        log_val_trans_counts.append(train_total[i])
    if train_total[i] != 0 and val_total[i] != 0:
        if val_err[i] != 0:
            log_val_err.append(val_err[i] / val_total[i])
        else:
            log_val_err.append(0)
        log_val_counts.append(train_total[i])

plt.plot(log_train_counts, log_train_err, 's', marker="o")
plt.plot(log_val_counts, log_val_err, 's', marker="s")
plt.plot(log_val_cis_counts, log_val_cis_err, 's', marker="v")
plt.plot(log_val_trans_counts, log_val_trans_err, 's', marker="x")

plt.xscale("symlog")
plt.yscale("symlog")
plt.title("Error Rate vs. Number of Training Examples Per Class")
plt.xlabel("Number of Training Examples For the Class")
plt.ylabel("Error Rate")
plt.legend(["Train", "Validation", "Validation (cis), Validation (trans)"])
plt.tight_layout()
plt.savefig("plots/error_v_num_ex_per_class.png")

plt.plot(log_train_counts, log_train_err, 's', marker="o")
plt.plot(log_val_counts, log_val_err, 's', marker="s")

plt.xscale("symlog")
plt.yscale("symlog")
plt.title("Error Rate vs. Number of Training Examples Per Class")
plt.xlabel("Number of Training Examples For the Class")
plt.ylabel("Error Rate")
plt.legend(["Train", "Validation"])
plt.tight_layout()
plt.savefig("plots/error_v_num_ex_per_class_general.png")

# Plot 9 mistakes.
train_mistakes = plot_mistakes(model, device, train_loader, 'plots/mistakes_train.png')
val_cis_mistakes = plot_mistakes(model, device, val_cis_loader, 'plots/mistakes_val_cis.png')
val_trans_mistakes = plot_mistakes(model, device, val_cis_loader, 'plots/mistakes_val_trans.png')
print('Train mistakes:', train_mistakes)
print('Val mistakes:', val_mistakes)
