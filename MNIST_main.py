import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from utils.aggregate import aggregate, generate_graph
from utils.plotting import plot_losses, plot_accuracies
from utils.earlystopping import EarlyStopper
from models.autoencoder import Encoder, Decoder
from models.linear_classifier import LinearClassifier
from tqdm import tqdm
import numpy as np

DEVICE = 'cuda:0'
N_WORKERS = 5

def MNIST_main(args):
    adj_matrix = generate_graph(N_WORKERS)
    if args.model=='autoencoder':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor()
        ])

        if args.model_training=='collaborative':
            encoders, _, losses, encoded_dim = train_loop(args.model, args.model_training, args.model_epochs, 16, train_transform, adj_matrix)
            plot_losses(losses, 'Collaborative_Autoencoder_MSE_Losses', args.output)
        
    if args.classifier=='linear':
        if args.classifier_training=='collaborative':
            classifiers, classifier_losses, classifier_accuracies = classifier_training(encoders, args.classifier_training, args.classifier_epochs, 16, encoded_dim, train_transform, adj_matrix)
            plot_losses(classifier_losses, 'Collaborative_Autoencoder_Classifier_Losses', args.output)
            plot_accuracies(classifier_accuracies, 'Collaborative_Autoencoder_Classifier_Accuracies', args.output)
    
    if args.testing=='local':
        test_accuracies = test_classifier(encoders, classifiers, args.testing)
    
    with open(os.path.join(args.output, 'test_accuracies'), 'w') as f:
        for key in test_accuracies.keys():
            f.write('{},{:.2f}\n'.format(key, test_accuracies[key]))


def test_classifier(model, classifier, mode: str):
    if type(classifier)==list:
        classifiers = classifier
        encoders = model
    
    if mode=='local':
        for classifier in classifiers:
            classifier.eval()

        testloaders = prepare_dataloaders(mode, batch_size=8, train=False)
        worker_accuracies = {0: [], 1: [], 2: [], 3: [], 4: []}

        for k in range(N_WORKERS):
            total = 0
            correct = 0
            testloader = testloaders[k]
            with torch.no_grad():
                for (features, labels) in testloader:
                    features, labels = features.to(DEVICE), labels.to(DEVICE)
                    reps = encoders[k](features)
                    outputs = classifiers[k](reps)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted==labels).sum().item()
        
            worker_accuracies[k] = (correct/total)*100
        
        return worker_accuracies
                    

def train_loop(model: str, mode: str, epochs: int, batch_size: int, train_transform, adj_matrix, encoded_dim: int=128, lr: float=1e-3):
    #main training loop for encoding model
    es = EarlyStopper(min_delta=0.5)

    if model=='autoencoder' and mode=='collaborative':
        encoders = [Encoder(encoded_dim).to(DEVICE) for k in range(N_WORKERS)]
        decoders = [Decoder(encoded_dim).to(DEVICE) for k in range(N_WORKERS)]
        trainloaders = prepare_dataloaders(mode, batch_size, train_transform)

        params_to_optimize = []
        for i in range(N_WORKERS):
            params_to_optimize.append([
                {'params': encoders[i].parameters()},
                {'params': decoders[i].parameters()}
            ])

        criterion = nn.MSELoss()
        optimizers = [torch.optim.Adam(params, lr=lr) for params in params_to_optimize]
        schedulers = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9) for optimizer in optimizers]

        worker_losses = {0: [], 1: [], 2: [], 3: [], 4: []}

        for i in range(N_WORKERS):
            encoders[i].train()
            decoders[i].train()

        for epoch in range(epochs):
            for k in range(N_WORKERS):
                curr_loss = []
                trainloader = trainloaders[k]
                for batch_idx, (features, _) in tqdm(enumerate(trainloader)):
                    features = features.to(DEVICE)

                    optimizers[k].zero_grad()
                    encoded = encoders[k](features)
                    decoded = decoders[k](encoded)

                    loss = criterion(decoded, features)
                    curr_loss.append(loss.item())
                    loss.backward()
                    optimizers[k].step()
                    schedulers[k].step()

                    if batch_idx%len(trainloader)==len(trainloader)-1:
                        avg_train_loss = np.mean(curr_loss)
                        print(f'In epoch {epoch} for worker {k}, average training loss is {avg_train_loss}.')
                        worker_losses[k].append(avg_train_loss)

            #check whether to stop early 
            curr_average = np.mean([worker_losses[k][epoch] for k in worker_losses.keys()])
            if es.early_stop(curr_average):
                print(f'Stopped training autoencoder after epoch {epoch}.')
                break
            #aggregate weights at the end of each epoch
            encoders = aggregate(N_WORKERS, encoders, adj_matrix)
            decoders = aggregate(N_WORKERS, decoders, adj_matrix)
        return encoders, decoders, worker_losses, encoded_dim
    

def classifier_training(encoder_model, mode: str, epochs: int, batch_size: int, encoded_dim: int, train_transform, adj_matrix, lr: float=1e-3):
    es = EarlyStopper(min_delta=0.2)
    
    if mode=='collaborative':
        #in this case encoder_model will be a list of encoders
        encoders = encoder_model
        classifiers = [LinearClassifier(encoded_dim, 10).to(DEVICE) for k in range(N_WORKERS)]
        optimizers = [torch.optim.Adam(classifier.parameters(), lr=lr) for classifier in classifiers]
        #schedulers = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9) for optimizer in optimizers]
        criterion = nn.CrossEntropyLoss()
        trainloaders = prepare_dataloaders(mode, batch_size, train_transform)

        for classifier in classifiers:
            classifier.train()

        classifier_accuracies = {0: [], 1: [], 2: [], 3: [], 4: []}
        classifier_losses = {0: [], 1: [], 2: [], 3: [], 4: []}

        for epoch in range(epochs):
            for k in range(N_WORKERS):
                total = 0
                correct = 0
                curr_loss = []
                trainloader = trainloaders[k]
                for batch_idx, (features, labels) in tqdm(enumerate(trainloader)):
                    features, labels = features.to(DEVICE), labels.to(DEVICE)

                    optimizers[k].zero_grad()
                    reps = encoders[k](features)
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

            classifiers = aggregate(N_WORKERS, classifiers, adj_matrix)

        return classifiers, classifier_losses, classifier_accuracies

def prepare_dataloaders(mode: str, batch_size: int, train_transform=None, train=True):
    test_transform = transforms.Compose([transforms.ToTensor()])
    worker_classes = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    if train==True:
        train_dataset = datasets.MNIST(root='./data', train=True, transform=train_transform, download=True)

        if mode=='collaborative':
            #separate data among workers by class
            worker_datasets = []
            for classes in worker_classes:
                idx = torch.cat([torch.where(train_dataset.targets==c)[0] for c in classes])
                worker_datasets.append(Subset(train_dataset, idx))

            trainloaders = []
            for i in range(len(worker_datasets)):
                trainloaders.append(DataLoader(worker_datasets[i], batch_size, shuffle=True, drop_last=True))

            return trainloaders
        
        else:
            return DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
    
    else:
        test_dataset = datasets.MNIST(root='./data', train=False, transform=test_transform, download=True)

        if mode=='local':
            test_datasets = []
            for classes in worker_classes:
                idx = torch.cat([torch.where(test_dataset.targets==c)[0] for c in classes])
                test_datasets.append(Subset(test_dataset, idx))
            
            testloaders = []
            for test_dataset in test_datasets:
                testloaders.append(DataLoader(test_dataset, batch_size, drop_last=True))

            return testloaders

        else:
            return DataLoader(test_dataset, batch_size, drop_last=True)
