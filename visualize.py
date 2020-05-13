import matplotlib.pyplot as plt
import numpy as np

def plot_loss_vs_size():
    sizes = np.log([0.001, 0.01, 0.1, 1])
    train_loss = np.log([2.1224, 2.4930, 1.5778, 1.1965])
    val_loss = np.log([5.4003, 3.9274, 2.8281, 2.4991])
    plt.plot(sizes, train_loss)
    plt.plot(sizes, val_loss)
    plt.xlabel("Log Scale Percentage of Training Data")
    plt.ylabel("Log Scale Cross Entropy Loss")
    plt.legend(["Train", "Validation"])
    plt.title("Loss vs. Training Set Size")
    plt.savefig("plots/loss_baseline.png")
    return

def plot_error_rate_vs_size():
    sizes = np.log([0.001, 0.01, 0.1, 1])
    train_loss = np.log([0.56, 0.65, 0.38, 0.32])
    val_loss = np.log([0.92, 0.82, 0.62, 0.56])
    plt.figure()
    plt.plot(sizes, train_loss)
    plt.plot(sizes, val_loss)
    plt.xlabel("Log Scale Percentage of Training Data")
    plt.ylabel("Log Scale Error Rate")
    plt.legend(["Train", "Validation"])
    plt.title("Error Rate vs. Training Set Size")
    plt.savefig("plots/error_baseline.png")
    return

if __name__ == "__main__":
    plot_loss_vs_size()
    plot_error_rate_vs_size()
