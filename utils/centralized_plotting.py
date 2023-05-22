import os
from matplotlib import pyplot as plt

def plot_losses(losses: list, title: str, output_dir=None):
    plt.clf()
    x = range(len(losses))
    plot_file = output_dir + '/' + title + '.png'
    csv_file = output_dir + '/' + title + '.csv'
    plt.plot(x, losses)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    if output_dir is not None:
        plt.savefig(plot_file)

        with open(csv_file, 'w', newline='') as f:
            f.write(str(loss) for loss in losses)
    else:
        plt.show()

def plot_accuracies(accuracies: list, title: str, output_dir=None):
    plt.clf()
    x = range(len(accuracies))
    plot_file = output_dir + '/' + title + '.png'
    csv_file = output_dir + '/' + title + '.csv'
    plt.plot(x, accuracies)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    if output_dir is not None:
        plt.savefig(plot_file)

        with open(csv_file, 'w', newline='') as f:
            f.write(str(acc) for acc in accuracies)
    else:
        plt.show()

def save_accuracy(accuracy, output):
    with open(os.path.join(output, 'test_accuracy'), 'w') as f:
        f.write(str(accuracy))

