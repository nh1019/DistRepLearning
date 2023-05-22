import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision.transforms import transforms

from models.autoencoder import Encoder
from models.linear_classifier import LinearClassifier
from utils.prepare_dataloaders import prepare_MNIST, prepare_CIFAR
from utils.earlystopping import EarlyStopper
from utils.save_config import save_config
from utils.centralized_plotting import *
from scripts.centralized_classifier import *

def main(args):
    save_config(args)

    encoder, classifier, _, classifier_losses, classifier_accuracies = train_EC(
        mode=args.model_training,
        dataset=args.dataset,
        batch_size=16,
        epochs=args.model_epochs,
        encoded_dim=args.encoded_dim)
    
    plot_losses(classifier_losses, f'{args.model_training}_Encoder_Classifier_Losses', args.output)
    plot_accuracies(classifier_accuracies, f'{args.model_training}_Encoder_Classifier_Accuracies', args.output)

    test_accuracy = test_classifier(encoder, classifier, args.dataset, args.testing)

    save_accuracy(test_accuracy, args.output)


def train_EC(mode: str, dataset: str, batch_size: int, epochs: int, encoded_dim: int, adj_matrix, lr: float=1e-3, device: str='cuda:0', n_workers: int=5):
    train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor()
        ])

    es = EarlyStopper(min_delta=0.5)

    if dataset=='MNIST':
        channels = 1
        trainloader = prepare_MNIST(mode, batch_size, train_transform)
    elif dataset=='CIFAR':
        channels = 3
        trainloader = prepare_CIFAR(mode, batch_size, train_transform)

    encoder = Encoder(channels, encoded_dim).to(device)
    classifier = LinearClassifier(encoded_dim, 10).to(device)
    criterion = nn.CrossEntropyLoss()

    params_to_optimize = [{'params': encoder.parameters()}, 
                          {'params': classifier.parameters()}]
    
    optimizer = torch.optim.Adam(params_to_optimize, lr=lr)

    classifier_losses = []
    classifier_accuracies = []

    for epoch in range(epochs):
        total = 0
        correct = 0
        curr_loss = []
        for batch_idx, data in tqdm(enumerate(trainloader)):
            features, labels = data
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            reps = encoder(features)
            classifier_output = classifier(reps)
            loss = criterion(classifier_output, labels)
            curr_loss.append(loss.item())
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(classifier_output.data, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()

            if batch_idx%len(trainloader)==len(trainloader)-1:
                avg_train_loss = np.mean(curr_loss)
                avg_train_acc = correct/total
                print(f'In epoch {epoch}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')
                classifier_losses.append(avg_train_loss)
                classifier_accuracies.append(avg_train_acc)

        if es.early_stop(avg_train_loss):
            print(f'Stopped training model after epoch {epoch}.')
            break

    return encoder, classifier, encoded_dim, classifier_losses, classifier_accuracies

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

    args = parser.parse_args()

    main(args)
