import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import torchvision.io as tvio
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

from models.linear_classifier import LinearClassifier
from utils.dist_plotting import plot_tsne
from utils.aggregate import aggregate
from utils.prepare_dataloaders import *
from utils.calc_norms import calculate_mean_norm


def train_classifier(models, 
                     dataset: str, 
                     mode: str, 
                     epochs: int, 
                     batch_size: int, 
                     encoded_dim: int, 
                     optimizer: str,
                     warmup_epochs: int,
                     scheduler: bool,
                     testing: str,
                     adj_matrix, 
                     data_fraction: float=1.,
                     train_transform=None, 
                     lr: float=1e-3, 
                     device: str='cuda:0', 
                     n_workers: int=5):
    
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.5,), (.5,))
        ])

    #es = EarlyStopper()

    if dataset=='MNIST':
        trainloaders = prepare_MNIST(mode, batch_size, train_transform, data_fraction=data_fraction)
    elif dataset=='CIFAR':
        trainloaders = prepare_CIFAR(mode, batch_size, train_transform)
        
    classifiers = [LinearClassifier(encoded_dim, 10).to(device) for _ in range(n_workers)]
    criterion = nn.CrossEntropyLoss()

    for i in range(n_workers):
        models[i].eval()
        classifiers[i].train()

    classifier_accuracies = {0: [], 1: [], 2: [], 3: [], 4: []}
    classifier_losses = {0: [], 1: [], 2: [], 3: [], 4: []}
    norms = []

    if warmup_epochs:
        desired_lr = lr
        initial_lr = desired_lr/100
    else:
        initial_lr = lr

    if optimizer=='Adam':
        optimizers = [torch.optim.Adam(classifier.parameters(), lr=initial_lr) for classifier in classifiers]
    elif optimizer=='SGD':
        optimizers = [torch.optim.SGD(classifier.parameters(), lr=initial_lr) for classifier in classifiers]
    elif optimizer=='AdamW':
        optimizers = [torch.optim.AdamW(classifier.parameters(), lr=initial_lr) for classifier in classifiers]
    else:
        raise ValueError('Please choose an implemented optimizer.')

    for epoch in range(warmup_epochs):
        current_lr = initial_lr + (desired_lr-initial_lr)*(epoch/warmup_epochs)
        for k in range(n_workers):
            trainloader = trainloaders[k]
            for param_group in optimizers[k].param_groups:
                param_group['lr'] = current_lr
            for batch_idx, (features, labels) in tqdm(enumerate(trainloader)):
                features, labels = features.to(device), labels.to(device)

                optimizers[k].zero_grad()
                reps = models[k](features)
                classifier_output = classifiers[k](reps)
                loss = criterion(classifier_output, labels)

                loss.backward()
                optimizers[k].step()

    if scheduler:
        schedulers = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, verbose=True) for optimizer in optimizers]

    for epoch in range(epochs):
        for k in range(n_workers):
            total = 0
            correct = 0
            curr_loss = []
            trainloader = trainloaders[k]
            for batch_idx, (features, labels) in tqdm(enumerate(trainloader)):
                features, labels = features.to(device), labels.to(device)

                optimizers[k].zero_grad()
                reps = models[k](features)
                classifier_output = classifiers[k](reps)
                loss = criterion(classifier_output, labels)

                curr_loss.append(loss.item())
                loss.backward()
                optimizers[k].step()

                #check prediction accuracy
                _, predicted = torch.max(classifier_output.data, 1)
                total += labels.size(0)
                correct += (predicted==labels).sum().item()

                if batch_idx%len(trainloader)==len(trainloader)-1:
                    avg_train_loss = np.mean(curr_loss)
                    avg_train_acc = (correct/total)*100
                    print(f'In epoch {epoch} for worker {k}, average training loss is {avg_train_loss}, average training accuracy is {avg_train_acc}%.')
                    classifier_losses[k].append(avg_train_loss)
                    classifier_accuracies[k].append(avg_train_acc)
        
        if scheduler:
            for k in range(n_workers):
                schedulers[k].step()

        if mode=='collaborative' and (testing=='global' or epoch<epochs-1):
            classifiers = aggregate(n_workers, classifiers, adj_matrix)
        
        norms.append(calculate_mean_norm(classifiers))

    return classifiers, classifier_losses, classifier_accuracies, norms

def test_classifier(models, 
                    classifier, 
                    dataset: str, 
                    mode: str, 
                    device: str='cuda:0', 
                    n_workers: int=5):
    
    classifiers = classifier
    
    for i in range(n_workers):
        models[i].eval()
        classifiers[i].eval()

    plot_tsne(models, dataset)

    if dataset=='MNIST':
        testloaders = prepare_MNIST(mode, batch_size=8, train=False)
    elif dataset=='CIFAR' and mode=='local':
        testloaders, test_datasets = prepare_CIFAR(mode, batch_size=8, train=False)
        #save examples
        for j in range(n_workers):
            img = test_datasets[j][0][0].cpu()
            encoded_img = models[j](img.unsqueeze(0).to(device)).detach().cpu().reshape(1, 1, -1)

            img = (img*255).to(torch.uint8)
            encoded_img = (encoded_img*255).to(torch.uint8)

            tvio.write_png(img, f'./results/original_{j}.png')
            tvio.write_png(encoded_img, f'./results/encoded_{j}.png')
    else:
        testloaders = prepare_CIFAR(mode, batch_size=8, train=False)


    worker_accuracies = {0: [], 1: [], 2: [], 3: [], 4: []}

    num_classes = 10
    confusion_matrices = []

    for k in range(n_workers):
        total = 0
        correct = 0
        testloader = testloaders[k] if mode=='local' else testloaders
        with torch.no_grad():
            true_labels = []
            predicted_labels = []
            for (features, labels) in testloader:
                features, labels = features.to(device), labels.to(device)
                reps = models[k](features)
                outputs = classifiers[k](reps)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted==labels).sum().item()

                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())

            cm = confusion_matrix(true_labels, predicted_labels, labels=range(num_classes))
            confusion_matrices.append(cm)
    
        worker_accuracies[k] = (correct/total)*100

    return worker_accuracies, confusion_matrices

def test_binary_classifier(model,
                           classifier,
                           dataset: str,
                           mode: str,
                           device: str='cuda:0',
                           n_workers: int=5):
    
    classifiers = classifier
    encoders = model

    for i in range(n_workers):
        encoders[i].eval()
        classifiers[i].eval()

    plot_tsne(model, dataset)

    activation = nn.Sigmoid()

    if dataset=='MNIST':
        testloaders = prepare_MNIST(mode, batch_size=8, train=False)
    else:
        testloaders, _ = prepare_CIFAR(mode, batch_size=8, train=False)
    
    worker_accuracies = {0: [], 1: [], 2: [], 3: [], 4: []}

    for k in range(n_workers):
        total = 0
        correct = 0
        testloader = testloaders[k]
        with torch.no_grad():
            for features, labels in testloader:
                labels = F.one_hot(torch.where(labels==2*k, torch.tensor(0), torch.tensor(1)))

                if labels.shape[1]==1:
                    #pad zeros
                    zeros_col = torch.zeros(labels.size(0), 1)
                    labels = torch.cat((labels, zeros_col), dim=1)
                    
                features, labels = features.to(device), labels.to(device)

                reps = encoders[k](features)
                outputs = activation(classifiers[k](reps))
                
                _, predicted = torch.max(outputs.data, 1)
                labels = torch.argmax(labels, dim=1)
                total += labels.size(0)
                correct += (predicted==labels).sum().item()

            worker_accuracies[k] = (correct/total)*100

    return worker_accuracies

