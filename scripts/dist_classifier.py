import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from utils.earlystopping import EarlyStopper
from models.linear_classifier import LinearClassifier
from utils.aggregate import aggregate
from utils.prepare_dataloaders import prepare_MNIST, prepare_CIFAR


def train_classifier(model, dataset: str, mode: str, epochs: int, batch_size: int, encoded_dim: int, train_transform, adj_matrix, lr: float=1e-3, device: str='cuda:0', n_workers: int=5):
    es = EarlyStopper(min_delta=0.2)

    if dataset=='MNIST':
        trainloaders = prepare_MNIST(mode, batch_size, train_transform)
    elif dataset=='CIFAR':
        trainloaders = prepare_CIFAR(mode, batch_size, train_transform)

    models = model
    classifiers = [LinearClassifier(encoded_dim, 10).to(device) for k in range(n_workers)]
    optimizers = [torch.optim.Adam(classifier.parameters(), lr=lr) for classifier in classifiers]
    #schedulers = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9) for optimizer in optimizers]
    criterion = nn.CrossEntropyLoss()

    for classifier in classifiers:
        classifier.train()

    classifier_accuracies = {0: [], 1: [], 2: [], 3: [], 4: []}
    classifier_losses = {0: [], 1: [], 2: [], 3: [], 4: []}

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
                #schedulers[k].step()

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
        
        curr_average = np.mean([classifier_losses[k][epoch] for k in classifier_losses.keys()])
        if es.early_stop(curr_average):
            print(f'Stopped training classifier after epoch {epoch}.')
            break

        if mode=='collaborative':
            classifiers = aggregate(n_workers, classifiers, adj_matrix)

    return classifiers, classifier_losses, classifier_accuracies