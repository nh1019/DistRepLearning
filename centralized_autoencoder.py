import argparse
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from tqdm import tqdm

from utils.prepare_dataloaders import prepare_CIFAR, prepare_MNIST
from models.autoencoder import Encoder, Decoder
from utils.earlystopping import EarlyStopper
from utils.save_config import save_config
from utils.centralized_plotting import *
from classifiers.centralized_classifier import *

def main(args):
    save_config(args)

    encoder, AE_losses, encoded_dim = train_AE(
        mode=args.model_training,
        dataset=args.dataset,
        batch_size=16,
        epochs=args.model_epochs,
        encoded_dim=args.encoded_dim,)
    
    plot_losses(AE_losses, f'{args.model_training}_Autoencoder_MSE_Losses', args.output)

    classifier, classifier_losses, classifier_accuracies = train_classifier(
        model=encoder,
        dataset=args.dataset,
        mode=args.classifier_training,
        epochs=args.classifier_epochs,
        batch_size=16,
        encoded_dim=encoded_dim)
    
    plot_losses(classifier_losses, f'Autoencoder_{args.classifier_training}_Classifier_Losses', args.output)
    plot_accuracies(classifier_accuracies, f'Autoencoder_{args.classifier_training}_Classifier_Accuracies', args.output)

    test_accuracies = test_classifier(
        model=encoder,
        classifier=classifier,
        dataset=args.dataset,
        mode=args.testing)
    
    save_accuracy(test_accuracies, args.output)


def train_AE(mode: str, dataset: str, batch_size: int, epochs: int, encoded_dim: int, lr: float=1e-4, device: str='cuda:0'):
    train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor()
        ])
    
    if dataset=='MNIST':
        channels = 1
        trainloader = prepare_MNIST(mode, batch_size, train_transform)
    elif dataset=='CIFAR':
        channels = 3
        trainloader = prepare_CIFAR(mode, batch_size, train_transform)

    epoch_losses = []
    encoder = Encoder(channels, encoded_dim).to(device)
    decoder = Decoder(channels, encoded_dim).to(device)
    
    es = EarlyStopper(min_delta=0.1)

    params_to_optimize = [{'params': encoder.parameters()}, 
                          {'params': decoder.parameters()}]

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    encoder.train()
    decoder.train()

    for epoch in range(epochs):
        curr_loss = []
        for batch_idx, (features, _) in tqdm(enumerate(trainloader)):
            features = features.to(device)

            optimizer.zero_grad()
            encoded = encoder(features)
            decoded = decoder(encoded)

            loss = criterion(decoded, features)
            curr_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            #scheduler.step()

            if batch_idx%len(trainloader)==len(trainloader)-1:
                avg_train_loss = np.mean(curr_loss)
                print(f'In epoch {epoch}, average training loss is {avg_train_loss}.')
                epoch_losses.append(avg_train_loss)

    #check whether to stop early 
        if es.early_stop(avg_train_loss):
            print(f'Stopped training autoencoder after epoch {epoch}.')
            break

    return encoder, epoch_losses, encoded_dim

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