from matplotlib import pyplot as plt
import os

def plot_losses(epochs: int, losses: dict, title: str, output_dir=None):
    x = range(epochs)
    savefile = output_dir + '/' + title + '.png'
    for key in losses.keys():
        plt.plot(x, losses[key], label='worker{}'.format(key))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    if output_dir is not None:
        #plt.savefig(os.path.join(output_dir, title, '.png'))
        plt.savefig(savefile)
    else:
        plt.show()

def plot_accuracies(epochs: int, accuracies: dict, title: str, output_dir=None):
    x = range(epochs)
    for key in accuracies.keys():
        plt.plot(x, accuracies[key], label='worker{}'.format(key))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, title, '.png'))
    else:
        plt.show()
