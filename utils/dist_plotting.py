import os
import csv
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from utils.prepare_dataloaders import *

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

def plot_confusion_matrix(cm, dataset, output_dir, worker):
    filename = output_dir + '/confusion_matrix_' + str(worker) + '.png'

    if dataset=='CIFAR':
        class_labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    else:
        class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    fig, ax = plt.subplots()
    heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title(f'Confusion Matrix Worker {worker}')
    ax.xaxis.set_ticklabels(class_labels)
    ax.yaxis.set_ticklabels(class_labels)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)

    plt.savefig(filename)


def save_accuracies(accuracies, output):
    with open(os.path.join(output, 'test_accuracies'), 'w') as f:
        for key in accuracies.keys():
            f.write('{},{:.2f}\n'.format(key, accuracies[key]))


'''
tsne plotting taken from https://medium.com/analytics-vidhya/understanding-simclr-a-simple-framework-for-contrastive-learning-of-visual-representations-d544a9003f3c
'''

def plot_vecs_n_labels(v, dataset, labels, fname):
    plt.axis('off')
    sns.set_style('darkgrid')
    sns.scatterplot(x=v[:,0], y=v[:,1], hue=labels, legend='full', palette=sns.color_palette('bright', 10))
    
    if dataset=='CIFAR':
        plt.legend(['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck'])
    else:
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    plt.savefig(fname)
    plt.close()

def plot_tsne(models, dataset, device='cuda:0'):
    tsne = TSNE()

    transform = transforms.ToTensor()
    
    if dataset=='CIFAR':
        loader = prepare_CIFAR('centralized', batch_size=500, train_transform=transform)
    else:
        loader = prepare_MNIST('centralized', batch_size=500, train_transform=transform)

    for i, model in enumerate(models):
        loader_iter = iter(loader)
        images, labels = next(loader_iter)
        reps = model(images.to(device))
        reps_tsne = tsne.fit_transform(reps.cpu().data)
        plot_vecs_n_labels(reps_tsne, dataset, labels, f'./results/t-SNE_worker_{i}')

        