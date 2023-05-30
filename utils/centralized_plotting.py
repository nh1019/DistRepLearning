import os
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from utils.prepare_dataloaders import *

def plot_losses(losses: list, title: str, output_dir=None):
    plt.clf()
    x = range(len(losses))
    plot_file = output_dir + '/' + title + '.png'
    csv_file = output_dir + '/' + title + '.csv'
    plt.plot(x, losses)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    if output_dir is not None:
        plt.savefig(plot_file)

        with open(csv_file, 'w', newline='') as f:
            for loss in losses:
                f.write(str(loss))
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

    if output_dir is not None:
        plt.savefig(plot_file)

        with open(csv_file, 'w', newline='') as f:
            for acc in accuracies:
                f.write(str(acc))
    else:
        plt.show()

def save_accuracy(accuracy, output):
    with open(os.path.join(output, 'test_accuracy'), 'w') as f:
        f.write(str(accuracy))


def plot_vecs_n_labels(v, dataset, labels, fname):
    plt.axis('off')
    sns.set_style('darkgrid')
    sns.scatterplot(x=v[:,0], y=v[:,1], hue=labels, style='dots', legend='full', palette=sns.color_palette('bright', 10))
    
    if dataset=='CIFAR':
        plt.legend(['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck'])
    else:
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    plt.savefig(fname)
    plt.close()

def plot_tsne(model, dataset, device='cuda:0'):
    tsne = TSNE()

    transform = transforms.ToTensor()
    
    if dataset=='CIFAR':
        loader = prepare_CIFAR('centralized', batch_size=500, train_transform=transform)
    else:
        loader = prepare_MNIST('centralized', batch_size=500, train_transform=transform)

    
    loader_iter = iter(loader)
    images, labels = next(loader_iter)
    reps = model(images.to(device))
    reps_tsne = tsne.fit_transform(reps.cpu().data)
    plot_vecs_n_labels(reps_tsne, dataset, labels, f'./results/t-SNE')

