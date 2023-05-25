import os
import csv
from matplotlib import pyplot as plt

def plot_losses(losses: dict, title: str, output_dir=None):
    plt.clf()
    x = range(len(losses[0]))
    plot_file = output_dir + '/' + title + '.png'
    csv_file = output_dir + '/' + title + '.csv'
    for key in losses.keys():
        plt.plot(x, losses[key], label='worker{}'.format(key))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    if output_dir is not None:
        plt.savefig(plot_file)

        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(losses.keys())
            writer.writerows(zip(*losses.values()))
    else:
        plt.show()

def plot_accuracies(accuracies: dict, title: str, output_dir=None):
    plt.clf()
    x = range(len(accuracies[0]))
    plot_file = output_dir + '/' + title + '.png'
    csv_file = output_dir + '/' + title + '.csv'
    for key in accuracies.keys():
        plt.plot(x, accuracies[key], label='worker{}'.format(key))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    if output_dir is not None:
        plt.savefig(plot_file)

        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(accuracies.keys())
            writer.writerows(zip(*accuracies.values()))
    else:
        plt.show()

def save_accuracies(accuracies, output):
    with open(os.path.join(output, 'test_accuracies'), 'w') as f:
        for key in accuracies.keys():
            f.write('{},{:.2f}\n'.format(key, accuracies[key]))