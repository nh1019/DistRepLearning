import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision.transforms import transforms

from models.autoencoder import Encoder
from models.linear_classifier import LinearClassifier
from utils.prepare_dataloaders import prepare_MNIST, prepare_CIFAR
from utils.aggregate import aggregate, generate_graph
from utils.save_config import save_config
from utils.dist_plotting import plot_losses, plot_accuracies, save_accuracies
from classifiers.dist_classifier import test_binary_classifier

def main(args):
    save_config(args)
    torch.manual_seed(2)
    np.random.seed(2)
    A = generate_graph(5, args.topology)

    fracs = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75]
    for frac in fracs:
        encoders, classifiers, _, classifier_losses, classifier_accuracies = train_EC(
            encoder_mode=args.model_training,
            classifier_mode=args.classifier_training,
            dataset=args.dataset,
            batch_size=16,
            epochs=args.model_epochs,
            encoded_dim=args.encoded_dim,
            adj_matrix=A,
            data_fraction=frac)
        
        plot_losses(classifier_losses, f'{args.model_training}_{frac}_Encoder_Binary_Classifier_Losses', args.output)
        plot_accuracies(classifier_accuracies, f'{args.model_training}_{frac}_Encoder_Binary_Classifier_Accuracies', args.output)

        test_accuracies = test_binary_classifier(encoders, 
                                        classifiers, 
                                        args.dataset, 
                                        args.testing)

        save_accuracies(test_accuracies, args.output, frac)


def train_EC(encoder_mode: str, 
             classifier_mode: str, 
             dataset: str, 
             batch_size: int, 
             epochs: int, 
             encoded_dim: int, 
             adj_matrix, 
             lr: float=1e-3,
             data_fraction: float=1., 
             device: str='cuda:0', 
             n_workers: int=5):
    
    train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor()
        ])

    if dataset=='MNIST':
        channels = 1
        trainloaders = prepare_MNIST(encoder_mode, batch_size, train_transform)
    elif dataset=='CIFAR':
        channels = 3
        trainloaders = prepare_CIFAR(mode=encoder_mode, batch_size=batch_size, train_transform=train_transform, data_fraction=data_fraction)

    encoders = [Encoder(channels, encoded_dim).to(device) for _ in range(n_workers)]
    classifiers = [LinearClassifier(encoded_dim, 2).to(device) for _ in range(n_workers)]
    criterion = nn.BCELoss()
    activation = nn.Sigmoid()

    params_to_optimize = []

    for i in range(n_workers):
        params_to_optimize.append([
            {'params': encoders[i].parameters()},
            {'params': classifiers[i].parameters()}
        ])
    
    optimizers = [torch.optim.AdamW(params, lr=lr) for params in params_to_optimize]
    schedulers = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, verbose=True, gamma=0.8) for optimizer in optimizers]

    for i in range(n_workers):
        encoders[i].train()
        classifiers[i].train()

    classifier_losses = {0: [], 1: [], 2: [], 3: [], 4: []}
    classifier_accuracies = {0: [], 1: [], 2: [], 3: [], 4: []}

    for epoch in range(epochs):
        for k in range(n_workers):
            total = 0
            correct = 0
            curr_loss = []
            trainloader = trainloaders[k]
            for batch_idx, data in tqdm(enumerate(trainloader)):
                features, labels = data
                #convert labels to 0s and 1s for binary classification
                labels = F.one_hot(torch.where(labels==2*k, torch.tensor(0), torch.tensor(1)))

                #check the case where the whole batch is equal to 2*k
                if labels.shape[1]==1:
                    #pad zeros
                    zeros_col = torch.zeros(labels.size(0), 1)
                    labels = torch.cat((labels, zeros_col), dim=1)
                features, labels = features.to(device), labels.to(device)

                optimizers[k].zero_grad()
                reps = encoders[k](features)
                classifier_output = activation(classifiers[k](reps))
                loss = criterion(classifier_output, labels.float())
                curr_loss.append(loss.item())
                loss.backward()
                optimizers[k].step()

                _, predicted = torch.max(classifier_output.data, 1)
                labels = torch.argmax(labels, dim=1)
                total += labels.size(0)
                correct += (predicted==labels).sum().item()

                if batch_idx%len(trainloader)==len(trainloader)-1:
                    avg_train_loss = np.mean(curr_loss)
                    avg_train_acc = correct/total
                    print(f'In epoch {epoch} for worker {k}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')
                    classifier_losses[k].append(avg_train_loss)
                    classifier_accuracies[k].append(avg_train_acc)

        for i in range(n_workers):
            schedulers[i].step()

        if encoder_mode=='collaborative':
            encoders = aggregate(n_workers, encoders, adj_matrix)
        
        if classifier_mode=='collaborative' and epoch!=epochs-1:
            classifiers = aggregate(n_workers, classifiers, adj_matrix)


    return encoders, classifiers, encoded_dim, classifier_losses, classifier_accuracies

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_training', type=str, help='choose between collaborative, non-collaborative, and centralized', required=True)
    parser.add_argument('--model_epochs', type=int, help='choose how many epochs to train the model for', required=True)
    parser.add_argument('--encoded_dim', type=int, help='specify dimension of encoded representations', required=True)
    parser.add_argument('--classifier_training', type=str, help='choose between collaborative and non-collaborative', required=True)
    parser.add_argument('--classifier_epochs', type=int, help='choose how many epochs to train the classifier for', required=True)
    parser.add_argument('--testing', type=str, help='choose between local and global (determines the data on which the classifier is tested)', required=True)
    parser.add_argument('--dataset', type=str, help='choose between MNIST and CIFAR (CIFAR-10)', required=True)
    parser.add_argument('--output', type=str, help='specify a folder for output files', required=True)
    parser.add_argument('--topology', type=str, help='choose a network topology to organise worker nodes', default='random')

    args = parser.parse_args()

    main(args)
