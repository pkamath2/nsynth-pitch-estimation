import matplotlib.pyplot as plt

def plot_losses(history_train, history_test, filename):
    
    fig, axes = plt.subplots(ncols=2, figsize=(10,10))
    for ax, metric in zip(axes, ['loss', 'acc']):
        ax.plot(history_train[metric])
        ax.plot(history_test[metric])
        ax.set_xlabel('epoch', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.legend(['Train', 'Test'], loc='best')
    plt.savefig(filename)
    plt.show()