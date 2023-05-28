import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix

from utils.earlystopping import EarlyStopper
from models.linear_classifier import LinearClassifier
from utils.aggregate import aggregate
from utils.prepare_dataloaders import prepare_MNIST, prepare_CIFAR


def train_classifier(model, dataset: str, mode: str, epochs: int, batch_size: int, encoded_dim: int, adj_matrix, train_transform=None, lr: float=1e-3, device: str='cuda:0', n_workers: int=5, simsiam=False):
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.5,), (.5,))
        ])

    es = EarlyStopper(min_delta=0.5)

    if dataset=='MNIST':
        trainloaders = prepare_MNIST(mode, batch_size, train_transform)
    elif dataset=='CIFAR':
        trainloaders = prepare_CIFAR(mode, batch_size, train_transform)

    if simsiam:
        models = [model.encoder for model in models]
    else:
        models = model
        
    classifiers = [LinearClassifier(encoded_dim, 10).to(device) for _ in range(n_workers)]
    criterion = nn.CrossEntropyLoss()

    for i in range(n_workers):
        models[i].eval()
        classifiers[i].train()

    classifier_accuracies = {0: [], 1: [], 2: [], 3: [], 4: []}
    classifier_losses = {0: [], 1: [], 2: [], 3: [], 4: []}

    warmup_epochs = 5
    initial_lr = 1e-5
    desired_lr = 1e-3

    optimizers = [torch.optim.SGD(classifier.parameters(), lr=initial_lr) for classifier in classifiers]
    schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3) for optimizer in optimizers]

    for epoch in range(warmup_epochs):
        current_lr = initial_lr + (desired_lr-initial_lr)*(epoch/warmup_epochs)
        for k in range(n_workers):
            trainloader = trainloaders[k]
            for param_group in optimizers[k].param_groups:
                param_group['lr'] = current_lr
            for batch_idx, (features, labels) in tqdm(enumerate(trainloader)):
                features, labels = features.to(device), labels.to(device)

                optimizers[k].zero_grad()
                encoded = models[k](features)
                classifier_output = classifiers[k](encoded)

                loss = criterion(classifier_output, labels)
                loss.backward()
                optimizers[k].step()

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
                schedulers[k].step(loss)

                #check prediction accuracy
                _, predicted = torch.max(classifier_output.data, 1)
                total += labels.size(0)
                correct += (predicted==labels).sum().item()

                if batch_idx%len(trainloader)==len(trainloader)-1:
                    print('true labels:')
                    print(labels)
                    print('predicted:')
                    print(predicted)
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

def test_classifier(model, classifier, dataset: str, mode: str, device: str='cuda:0', n_workers: int=5, simsiam=False):
    classifiers = classifier

    if simsiam:
        encoders = [encoder.encoder for encoder in model]
    else:
        encoders = model
    
    for i in range(n_workers):
        encoders[i].eval()
        classifiers[i].eval()

    if dataset=='MNIST':
        testloaders = prepare_MNIST(mode, batch_size=8, train=False)
    elif dataset=='CIFAR':
        testloaders, test_datasets = prepare_CIFAR(mode, batch_size=8, train=False)
        #save examples
        for j in range(n_workers):
            img = test_datasets[j][0][0].to(device)
            img = img.unsqueeze(0)
            encoded_img = encoders[j](img).cpu().detach().numpy()
            normalized = (encoded_img*256).astype('uint8')
            pil_img = Image.fromarray(normalized)
            pil_img.save(f'./results/example_{j}.png')


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
                reps = encoders[k](features)
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