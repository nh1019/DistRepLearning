import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision.transforms import transforms

from models.autoencoder import Encoder, Decoder
from utils.prepare_dataloaders import prepare_CIFAR
from utils.aggregate import aggregate, generate_graph
from utils.dist_plotting import *
from utils.save_config import save_config
from classifiers.dist_classifier import *

def main(args):
    save_config(args)
    torch.manual_seed(2)
    np.random.seed(2)
    A = generate_graph(5, args.topology)

    fracs = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75]
    for frac in fracs:
        encoders, AE_losses, encoded_dim = train_AE(
            mode=args.model_training,
            batch_size=16,
            epochs=args.model_epochs,
            encoded_dim=args.encoded_dim,
            optimizer=args.optimizer,
            warmup_epochs=args.warmup_epochs,
            scheduler=args.scheduler,
            adj_matrix=A,
            data_fraction=frac)
        
        plot_losses(AE_losses, f'{args.model_training}_frac_Autoencoder_MSE_Losses', args.output)

        classifiers, classifier_losses, classifier_accuracies = train_classifier(
            models=encoders,
            dataset=args.dataset,
            mode=args.classifier_training,
            epochs=args.classifier_epochs,
            batch_size=16,
            encoded_dim=encoded_dim,
            optimizer=args.optimizer,
            warmup_epochs=args.warmup_epochs,
            scheduler=args.scheduler,
            testing=args.testing,
            data_fraction=frac,
            adj_matrix=A)
        
        plot_losses(classifier_losses, f'Autoencoder_{args.classifier_training}_frac_Classifier_Losses', args.output)
        plot_accuracies(classifier_accuracies, f'Autoencoder_{args.classifier_training}_frac_Classifier_Accuracies', args.output)

        test_accuracies, confusion_matrices = test_classifier(
            models=encoders,
            classifier=classifiers,
            dataset=args.dataset,
            mode=args.testing)
        
        for i, cm in enumerate(confusion_matrices):
            plot_confusion_matrix(cm, args.dataset, args.output, i)
        
        save_accuracies(test_accuracies, args.output)

def train_AE(mode: str,  
             batch_size: int, 
             epochs: int, 
             encoded_dim: int, 
             optimizer: str,
             warmup_epochs: int,
             scheduler: bool,
             adj_matrix,
             lr: float=5e-3,
             data_fraction: float=1., 
             device: str='cuda:0', 
             n_workers: int=5):
    
    train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor()
        ])
    
    trainloaders = prepare_CIFAR(mode=mode, batch_size=batch_size, train_transform=train_transform, train=True, iid=True, data_fraction=data_fraction)
    channels=3

    worker_losses = {0: [], 1: [], 2: [], 3: [], 4: []}
    encoders = [Encoder(channels, encoded_dim).to(device) for _ in range(n_workers)]
    decoders = [Decoder(channels, encoded_dim).to(device) for _ in range(n_workers)]
    
    #es = EarlyStopper()

    params_to_optimize = []
    for i in range(n_workers):
        params_to_optimize.append([
            {'params': encoders[i].parameters()},
            {'params': decoders[i].parameters()}
        ])

    if warmup_epochs:
        desired_lr = lr
        initial_lr = desired_lr/100
    else:
        initial_lr = lr

    criterion = nn.MSELoss()

    if optimizer=='Adam':
        optimizers = [torch.optim.Adam(params, lr=initial_lr) for params in params_to_optimize]
    elif optimizer=='SGD':
        optimizers = [torch.optim.SGD(params, lr=initial_lr) for params in params_to_optimize]
    elif optimizer=='AdamW':
        optimizers = [torch.optim.AdamW(params, lr=initial_lr) for params in params_to_optimize]
    else:
        raise ValueError('Please choose an implemented optimizer.')

    for i in range(n_workers):
        encoders[i].train()
        decoders[i].train()

    for epoch in range(warmup_epochs):
        current_lr = initial_lr + (desired_lr-initial_lr)*(epoch/warmup_epochs)
        for k in range(n_workers):
            trainloader = trainloaders[k]
            for param_group in optimizers[k].param_groups:
                param_group['lr'] = current_lr
            for batch_idx, (features, _) in tqdm(enumerate(trainloader)):
                features = features.to(device)

                optimizers[k].zero_grad()
                encoded = encoders[k](features)
                decoded = decoders[k](encoded)

                loss = criterion(decoded, features)
                loss.backward()
                optimizers[k].step()

    if scheduler:
        schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True) for optimizer in optimizers]

    for epoch in range(epochs):
        for k in range(n_workers):
            curr_loss = []
            trainloader = trainloaders[k]
            for batch_idx, (features, _) in tqdm(enumerate(trainloader)):
                features = features.to(device)

                optimizers[k].zero_grad()
                encoded = encoders[k](features)
                decoded = decoders[k](encoded)

                loss = criterion(decoded, features)
                curr_loss.append(loss.item())
                loss.backward()
                optimizers[k].step()

                if batch_idx%len(trainloader)==len(trainloader)-1:
                    avg_train_loss = np.mean(curr_loss)
                    print(f'In epoch {epoch} for worker {k}, average training loss is {avg_train_loss}.')
                    worker_losses[k].append(avg_train_loss)

        
        #check whether to stop early 
        curr_average = np.mean([worker_losses[k][epoch] for k in worker_losses.keys()])
        if scheduler:
            for k in range(n_workers):
                schedulers[k].step(curr_average)

        #aggregate weights at the end of each epoch
        if mode=='collaborative':
            encoders = aggregate(n_workers, encoders, adj_matrix)
            decoders = aggregate(n_workers, decoders, adj_matrix)

    return encoders, worker_losses, encoded_dim

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
    parser.add_argument('--topology', type=str, help='choose a network topology to organise worker nodes', default='random')
    parser.add_argument('--data_fraction', type=float, default=1., help='to test with limited dataset size. pick a value between 0 and 1.')

    args = parser.parse_args()

    main(args)
