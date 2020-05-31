import torch
import numpy as np
import time

# https://github.com/adambielski/siamese-triplet/blob/master/trainer.py

def fit(train_loader, val_cis_loader, val_trans_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[]):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model
    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    train_losses = []
    val_cis_losses = []
    val_trans_losses = []
    start = time.time()
    for epoch in range(1, n_epochs+1):
        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_cis_loss, metrics = test_epoch(val_cis_loader, model, loss_fn, cuda, metrics)
        val_cis_loss /= len(val_cis_loader)

        message += '\nEpoch: {}/{}. Validation set (cis): Average loss: {:.4f}'.format(epoch, n_epochs,
                                                                                 val_cis_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_trans_loss, metrics = test_epoch(val_trans_loader, model, loss_fn, cuda, metrics)
        val_trans_loss /= len(val_trans_loader)

        message += '\nEpoch: {}/{}. Validation set (trans): Average loss: {:.4f}'.format(epoch, n_epochs,
                                                                                 val_trans_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        train_losses.append(train_loss)
        val_cis_losses.append(val_cis_loss)
        val_trans_losses.append(val_trans_loss)
        scheduler.step()
        torch.save(model.state_dict(), "triplet_batch_all_{}.pt".format(epoch))

        print(message)

    print('Train Time:', time.time()-start)
    # You may optionally save your model at each epoch here
    np.save("train_loss_batch_all.npy", np.array(train_losses))
    np.save("test_cis_loss_batch_all.npy", np.array(val_cis_losses))
    np.save("test_trans_loss_batch_all.npy", np.array(val_trans_losses))



def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()


        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics
