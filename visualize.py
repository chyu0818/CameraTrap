import matplotlib.pyplot as plt
import numpy as np

def plot_loss_vs_size():
    sizes = [0.001, 0.01, 0.1, 1]
    # train_loss = [2.1224, 2.4930, 1.5778, 1.1965]
    # val_loss = [5.4003, 3.9274, 2.8281, 2.4991]
    train_loss = [0.0173, 0.1735, 0.0909, 0.0342]
    val_loss = [4.3461, 3.0954, 2.1724, 2.1016]
    fig, ax = plt.subplots()
    ax.plot(sizes, train_loss, marker='o', label='Train')
    ax.plot(sizes, val_loss, marker='o', label='Validation')
    ax.set(xlabel='Percentage of Training Data',
           ylabel='Cross Entropy Loss',
           xscale='log', yscale='log',
           title='Loss vs. Training Set Size')
    ax.legend()
    plt.tight_layout()
    plt.savefig("plots/loss_baseline.png")
    plt.show()
    return
# do again but with full validation set
def plot_error_rate_vs_size():
    sizes = [0.001, 0.01, 0.1, 1]
    train_loss = [0.000001, 0.02, 0.01, 0.02]
    val_loss = [0.76, 0.66, 0.45, 0.36]
    fig, ax = plt.subplots()
    ax.plot(sizes, train_loss, marker='o', label='Train')
    ax.plot(sizes, val_loss, marker='o', label='Validation')
    ax.set(xlabel='Percentage of Training Data',
           ylabel='Error Rate',
           xscale='log', yscale='log',
           title='Error Rate vs. Training Set Size')
    ax.legend()
    plt.tight_layout()
    plt.savefig("plots/error_baseline.png")
    plt.show()
    return

# Plots training and val loss as a function of the epoch.
def plot_train_test_loss_epoch(train_loss, test_loss, test_cis_loss, test_trans_loss, num_epochs):

    fig, ax = plt.subplots()
    epochs = list(range(1,num_epochs+1))
    ax.plot(epochs, train_loss, color='r', marker='.', label='Train')
    ax.plot(epochs, test_loss, color='b', marker='.', label='Validation')
    ax.plot(epochs, test_cis_loss, color='g', marker='.', label='Validation (Cis)')
    ax.plot(epochs, test_trans_loss, color='c', marker='.', label='Validation (Trans)')
    ax.set(xlabel='Epoch', ylabel='Loss', yscale='log', title='Loss By Epoch')
    ax.legend(title='Type of Loss')
    plt.tight_layout()
    plt.savefig('loss_by_epoch_64.png')

    fig, ax = plt.subplots()
    epochs = list(range(1,num_epochs+1))
    ax.plot(epochs, test_loss, color='b', marker='.', label='Validation')
    ax.plot(epochs, test_cis_loss, color='g', marker='.', label='Validation (Cis)')
    ax.plot(epochs, test_trans_loss, color='c', marker='.', label='Validation (Trans)')
    ax.set(xlabel='Epoch', ylabel='Loss', yscale='log', title='Loss By Epoch')
    ax.legend(title='Type of Loss (only validation)')
    plt.tight_layout()
    plt.savefig('loss_by_epoch_val_64.png')
    return

def main():
    # plot_loss_vs_size()
    # plot_error_rate_vs_size()
    num_val_cis = 3307
    num_val_trans = 6382
    num_val_cis_frac = num_val_cis / (num_val_cis + num_val_trans)
    num_val_trans_frac = num_val_trans/ (num_val_cis + num_val_trans)

    train_loss = np.load('train_loss.npy')
    test_cis_loss = np.load('test_cis_loss.npy')
    test_trans_loss = np.load('test_trans_loss.npy')
    print(train_loss)
    print(test_cis_loss)
    print(test_trans_loss)
    test_loss = num_val_cis_frac * test_cis_loss + num_val_trans_frac * test_trans_loss
    plot_train_test_loss_epoch(train_loss, test_loss, test_cis_loss, test_trans_loss, 20)

if __name__ == "__main__":
    main()
