import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import numpy as np

from utils.earlystopping import EarlyStopper
from models.linear_classifier import LinearClassifier
from utils.prepare_dataloaders import prepare_MNIST, prepare_CIFAR
from utils.centralized_plotting import plot_tsne


def train_classifier(model, 
                     dataset: str, 
                     mode: str, 
                     epochs: int, 
                     batch_size: int, 
                     encoded_dim: int, 
                     optimizer: str,
                     warmup_epochs: int,
                     scheduler: bool,
                     train_transform=None, 
                     lr: float=1e-3, 
                     device: str='cuda:0', 
                     simsiam=False):
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.5,), (.5,))
        ])

    #es = EarlyStopper(min_delta=0.2)

    if dataset=='MNIST':
        trainloader = prepare_MNIST(mode, batch_size, train_transform)
    elif dataset=='CIFAR':
        trainloader = prepare_CIFAR(mode, batch_size, train_transform)

    if simsiam:
        encoder = model.encoder
    else:
        encoder = model

    if warmup_epochs:
        desired_lr = lr
        initial_lr = desired_lr/100
    else:
        initial_lr = lr
        
    classifier = LinearClassifier(encoded_dim, 10).to(device)

    if optimizer=='Adam':
        optim = torch.optim.Adam(classifier.parameters(), lr=initial_lr)
    elif optimizer=='AdamW':
        optim = torch.optim.AdamW(classifier.parameters(), lr=initial_lr)
    elif optimizer=='SGD':
        optim = torch.optim.SGD(classifier.parameters(), lr=initial_lr)
    else:
        raise ValueError('Please choose an implemented optimizer.')

    if scheduler:
        sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

    criterion = nn.CrossEntropyLoss()

    encoder.eval()
    classifier.train()

    for epoch in range(warmup_epochs):
        current_lr = initial_lr + (desired_lr-initial_lr)*(epoch/warmup_epochs)
        for param_group in optim.param_groups:
            param_group['lr'] = current_lr
        for batch_idx, (features, labels) in tqdm(enumerate(trainloader)):
            features, labels = features.to(device), labels.to(device)

            optim.zero_grad()
            reps = encoder(features)
            classifier_output = classifier(reps)
            loss = criterion(classifier_output, labels)

            loss.backward()
            optim.step()

    classifier_accuracies = []
    classifier_losses = []

    for epoch in range(epochs):
        total = 0
        correct = 0
        curr_loss = []
        for batch_idx, (features, labels) in tqdm(enumerate(trainloader)):
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            reps = encoder(features)
            classifier_output = classifier(reps)
            loss = criterion(classifier_output, labels)

            curr_loss.append(loss.item())
            loss.backward()
            optimizer.step()

            if scheduler:
                sched.step()

            #check prediction accuracy
            _, predicted = torch.max(classifier_output.data, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()

            if batch_idx%len(trainloader)==len(trainloader)-1:
                avg_train_loss = np.mean(curr_loss)
                avg_train_acc = (correct/total)*100
                print(f'In epoch {epoch}, average training loss is {avg_train_loss}, average training accuracy is {avg_train_acc}%.')
                classifier_losses.append(avg_train_loss)
                classifier_accuracies.append(avg_train_acc)

        '''
        if es.early_stop(avg_train_loss):
            print(f'Stopped training classifier after epoch {epoch}.')
            break
        '''

    return classifier, classifier_losses, classifier_accuracies

def test_classifier(model, classifier, dataset: str, mode: str, device: str='cuda:0', simsiam=False):
    if simsiam:
        encoder = model.encoder
    else:
        encoder = model
    
    encoder.eval()
    classifier.eval()

    plot_tsne(encoder, dataset)

    if dataset=='MNIST':
        testloader = prepare_MNIST(mode, batch_size=8, train=False)
    elif dataset=='CIFAR':
        testloader = prepare_CIFAR(mode, batch_size=8, train=False)

    total = 0
    correct = 0
    testloader = testloader
    with torch.no_grad():
        for (features, labels) in testloader:
            features, labels = features.to(device), labels.to(device)
            reps = encoder(features)
            outputs = classifier(reps)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()

    accuracy = (correct/total)*100
    
    return accuracy