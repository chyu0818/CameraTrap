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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from triplet_loss import TripletNet, TripletLoss, Embedder, TripletLossBatchAll, TripletLossBatchHard

BATCH_SIZE_TRAIN = 512
BATCH_SIZE_VAL = 512
LOG_INTERVAL = 10
NUM_CLASSES = 267 # 267??
NUM_EPOCHS = 20
NUM_VAL_CIS = 3307
NUM_VAL_TRANS = 6382
NUM_VAL_CIS_FRAC = NUM_VAL_CIS / (NUM_VAL_CIS + NUM_VAL_TRANS)
NUM_VAL_TRANS_FRAC = NUM_VAL_TRANS/ (NUM_VAL_CIS + NUM_VAL_TRANS)

# top 5?
def test(model, device, test_loader):
    criterion = nn.CrossEntropyLoss(reduction='sum')
    model.eval()    # Set the model to inference mode
    error = np.zeros(267)
    total = np.zeros(267)
    test_loss = 0
    correct = 0
    total1 = 0
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data0 in test_loader:
            data = data0['image']
            target = data0['target']
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total1 += len(target)
            for i in range(len(pred)):
                if pred[i,0] != target[i]:
                    error[target[i]] += 1
                total[target[i]] += 1
    test_loss /= total1
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total1,
        100. * correct / total1))
    return error, total, test_loss


def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for data0 in dataloader:
            images = data0['image']
            target = data0['target']
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels


# Visualizes the tSNE embedding.
def plot_embeddings(embeddings, targets, classes, title):
    colors = cm.rainbow(np.linspace(0, 1, len(classes)))
    embeddings_pca = PCA(n_components=200).fit_transform(embeddings)
    embeddings_tsne = TSNE(n_components=2).fit_transform(embeddings_pca)

    for j in classes:
        inds = np.where(targets==j)[0]
        plt.scatter(embeddings_tsne[inds,0], embeddings_tsne[inds,1],
                    color=colors[j], marker='.', alpha=0.5, label=j)
    plt.legend(title='Class')
    plt.title('Feature Vectors By Class ({})'.format(title))
    plt.savefig('tSNE_embedding_{}.png'.format(title))
    return embeddings_tsne


# Plots 9 examples from the test set where the classifier made a mistake.
def plot_mistakes(model, device, test_loader, save_fn):
    annotations_fn = '../efs/iwildcam2020_train_annotations.json'
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
                    ax.set_title('Actual: {} Pred: {}'.format(target[i], pred[i,0]))
                    print('Actual:', target[i], all_categories[target[i]]['name'], 'Pred:', pred[i,0], all_categories[pred[i,0]]['name'])
                    if len(mistakes) >= lim_mistakes:
                        plt.tight_layout()
                        plt.savefig(save_fn)
                        return mistakes
    return

def plot_error_rate_by_num_ex_class(model, device, train_loader, val_cis_loader, val_trans_loader):
    print('Training set:')
    train_err, train_total, train_loss = test(model, device, train_loader)
    print('Validation set (cis):')
    val_cis_err, val_cis_total, val_cis_loss = test(model, device, val_cis_loader)
    print('Validation set (trans):')
    val_trans_err, val_trans_total, val_trans_loss = test(model, device, val_trans_loader)
    val_err = val_cis_err + val_trans_err
    val_total = val_cis_total + val_trans_total
    val_loss = NUM_VAL_CIS_FRAC * val_cis_loss + NUM_VAL_TRANS_FRAC * val_trans_loss
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

    fig, ax = plt.subplots()
    ax.plot(log_train_counts, log_train_err, color='r', marker='o', label='Train', linestyle='None')
    ax.plot(log_val_counts, log_val_err, color='b', marker='s', label='Validation', linestyle='None')
    ax.set(xlabel='Number of Training Examples For the Class', ylabel='Error Rate', 
           xscale='symlog', yscale='symlog', 
           title='Error Rate vs. Number of Training Examples Per Class')
    ax.legend()
    plt.tight_layout()
    plt.savefig('plots/error_v_num_ex_per_class_general.png')

    fig, ax = plt.subplots()
    ax.plot(log_val_cis_counts, log_val_cis_err, color='g', marker='v', label='Validation (Cis)', linestyle='None')
    ax.plot(log_val_trans_counts, log_val_trans_err, color='c', marker='x', label='Validation (Trans)', linestyle='None')
    ax.set(xlabel='Number of Training Examples For the Class', ylabel='Error Rate', 
           xscale='symlog', yscale='symlog', 
           title='Error Rate vs. Number of Training Examples Per Class')
    ax.legend()
    plt.tight_layout()
    plt.savefig('plots/error_v_num_ex_per_class.png')
    return


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}
    # model = models.resnet18(pretrained=True)

    # model.fc = torch.nn.Linear(512, NUM_CLASSES)
    # model.to(device)

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    #transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    # Not sure if Crop is required
    transform = T.Compose([T.Resize((256,256)), T.ToTensor(), normalize])

    train_path = 'X_train.npz'
    val_cis_path = 'X_val_cis.npz'
    val_trans_path = 'X_val_trans.npz'
    img_path = '../efs/train_crop'
    ann_path = '../efs/iwildcam2020_train_annotations.json'
    bbox_path = '../efs/iwildcam2020_megadetector_results.json'
    model_path = "models/triplet_batch_all_2_conf_9.pt"
    percent_data = 0.01
    # ~70k train, ~20k val


    embedding_net = models.resnet18(pretrained=True)
    # for param in embedding_net.parameters():
    #     param.requires_grad = False
    embedding_net.fc = torch.nn.Linear(512, 1000)

    # model = TripletNet(embedding_net)
    model = Embedder(embedding_net)
    model.cuda()

    model.load_state_dict(torch.load(model_path))

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

    # Choose 10 random classes out of classes with over 1000 (not empty or human)
    with open(ann_path) as f:
        annotations = json.load(f)
    categories = annotations['categories']
    categories_sort = sorted(categories, key=lambda k: k['count'], reverse=True)
    categories_dict = {categories[i]['id']:i for i in range(len(categories))}
    names = []
    ids = []
    counts = []
    for i in range(len(categories)):
        names.append(categories_sort[i]['name'])
        ids.append(categories_dict[categories_sort[i]['id']])
        counts.append(categories_sort[i]['count'])
    names = np.asarray(names)
    ids = np.asarray(ids)
    print(ids)
    counts = np.asarray(counts)

    # Find classes over 1000. 
    classes_all = ids[counts >= 1000]
    classes_rand = np.random.choice(classes_all, 10)
    print(classes_all, len(classes_all))
    print(classes_rand)

    # Plot embeddings
    val_cis_embeddings, val_cis_targets = extract_embeddings(val_cis_loader, model)
    plot_embeddings(val_cis_embeddings, val_cis_targets, classes_rand, 'val_cis')

    val_trans_embeddings, val_trans_targets = extract_embeddings(val_cis_loader, model)
    plot_embeddings(val_trans_embeddings, val_trans_targets, classes_rand, 'val_trans')

    train_embeddings, train_targets = extract_embeddings(train_loader, model)
    plot_embeddings(train_embeddings, train_targets, classes_rand, 'train')


    # Plot error rate by number of examples in class.
    # plot_error_rate_by_num_ex_class(model, device, train_loader, val_cis_loader, val_trans_loader)

    # Plot 9 mistakes.
    # train_mistakes = plot_mistakes(model, device, train_loader, 'plots/mistakes_train.png')
    # val_cis_mistakes = plot_mistakes(model, device, val_cis_loader, 'plots/mistakes_val_cis.png')
    # val_trans_mistakes = plot_mistakes(model, device, val_cis_loader, 'plots/mistakes_val_trans.png')
    # print('Train mistakes:', train_mistakes)
    # print('Val cis mistakes:', val_cis_mistakes)
    # print('Val trans mistakes:', val_trans_mistakes)

if __name__ == '__main__':
    main()
