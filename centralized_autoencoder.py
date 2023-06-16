import argparse
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from tqdm import tqdm

from utils.prepare_dataloaders import prepare_CIFAR, prepare_MNIST
from models.autoencoder import Encoder, Decoder
from utils.save_config import save_config
from utils.centralized_plotting import *
from classifiers.centralized_classifier import *

def main(args):
    save_config(args)

    fracs = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75]
    for frac in fracs:
        encoder, AE_losses, encoded_dim = train_AE(
            mode=args.model_training,
            dataset=args.dataset,
            batch_size=16,
            epochs=args.model_epochs,
            encoded_dim=args.encoded_dim,
            optimizer=args.optimizer,
            data_fraction=frac,
            warmup_epochs=args.warmup_epochs,
            scheduler=args.scheduler)
        
        plot_losses(AE_losses, f'{args.model_training}_{frac}_Autoencoder_MSE_Losses', args.output)

        classifier, classifier_losses, classifier_accuracies = train_classifier(
            model=encoder,
            dataset=args.dataset,
            mode=args.classifier_training,
            epochs=args.classifier_epochs,
            batch_size=16,
            encoded_dim=encoded_dim,
            optimizer=args.optimizer,
            data_fraction=frac,
            warmup_epochs=args.warmup_epochs,
            scheduler=args.scheduler)
        
        plot_losses(classifier_losses, f'Autoencoder_{args.classifier_training}_{frac}_Classifier_Losses', args.output)
        plot_accuracies(classifier_accuracies, f'Autoencoder_{args.classifier_training}_{frac}_Classifier_Accuracies', args.output)

        test_accuracies = test_classifier(
            model=encoder,
            classifier=classifier,
            dataset=args.dataset,
            mode=args.testing)
        
        save_accuracy(test_accuracies, args.output, frac)


def train_AE(mode: str, 
             dataset: str, 
             batch_size: int, 
             epochs: int, 
             encoded_dim: int, 
             optimizer: str, 
             warmup_epochs: int, 
             scheduler: bool,
             data_fraction: float=1.,
             lr: float=1e-4, 
             device: str='cuda:0'):
    
    train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor()
        ])
    
    if dataset=='MNIST':
        channels = 1
        trainloader = prepare_MNIST(mode=mode, batch_size=batch_size, train_transform=train_transform, data_fraction=data_fraction)
    elif dataset=='CIFAR':
        channels = 3
        trainloader = prepare_CIFAR(mode=mode, batch_size=batch_size, train_transform=train_transform, data_fraction=data_fraction)

    epoch_losses = []
    encoder = Encoder(channels, encoded_dim).to(device)
    decoder = Decoder(channels, encoded_dim).to(device)
    
    #es = EarlyStopper(min_delta=0.1)

    params_to_optimize = [{'params': encoder.parameters()}, 
                          {'params': decoder.parameters()}]

    if warmup_epochs:
        desired_lr = lr
        initial_lr = desired_lr/100
    else:
        initial_lr = lr

    criterion = nn.MSELoss()

    if optimizer=='Adam':
        optim = torch.optim.Adam(params_to_optimize, lr=initial_lr)
    elif optimizer=='AdamW':
        optim = torch.optim.AdamW(params_to_optimize, lr=initial_lr)
    elif optimizer=='SGD':
        optim = torch.optim.SGD(params_to_optimize, lr=initial_lr)
    else:
        raise ValueError('Please choose an implemented optimizer.')

    encoder.train()
    decoder.train()

    for epoch in range(warmup_epochs):
        current_lr = initial_lr + (desired_lr-initial_lr)*(epoch/warmup_epochs)
        for param_group in optim.param_groups:
            param_group['lr'] = current_lr
        for batch_idx, (features, _) in tqdm(enumerate(trainloader)):
            features = features.to(device)

            optim.zero_grad()
            encoded = encoder(features)
            decoded = decoder(encoded)

            loss = criterion(decoded, features)
            loss.backward()
            optim.step()

    if scheduler:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=5, factor=0.5, verbose=True)

    for epoch in range(epochs):
        curr_loss = []
        for batch_idx, (features, _) in tqdm(enumerate(trainloader)):
            features = features.to(device)

            optim.zero_grad()
            encoded = encoder(features)
            decoded = decoder(encoded)

            loss = criterion(decoded, features)
            curr_loss.append(loss.item())
            loss.backward()
            optim.step()

            if batch_idx%len(trainloader)==len(trainloader)-1:
                avg_train_loss = np.mean(curr_loss)
                print(f'In epoch {epoch}, average training loss is {avg_train_loss}.')
                epoch_losses.append(avg_train_loss)

        if scheduler:
                sched.step(avg_train_loss)

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
    parser.add_argument('--optimizer', type=str, help='choose between SGD, Adam, and AdamW', required=True)
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--scheduler', action='store_true', default=False)

    args = parser.parse_args()

    main(args)