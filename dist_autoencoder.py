import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision.transforms import transforms

from models.autoencoder import Encoder, Decoder
from utils.prepare_dataloaders import prepare_MNIST, prepare_CIFAR
from utils.aggregate import aggregate, generate_graph
from utils.earlystopping import EarlyStopper
from utils.dist_plotting import *
from utils.save_config import save_config
from classifiers.dist_classifier import *

def main(args):
    save_config(args)
    torch.manual_seed(2)
    np.random.seed(2)
    A = generate_graph(5, args.topology)

    encoders, AE_losses, encoded_dim = train_AE(
        mode=args.model_training,
        dataset=args.dataset,
        batch_size=16,
        epochs=args.model_epochs,
        encoded_dim=args.encoded_dim,
        adj_matrix=A)
    
    plot_losses(AE_losses, f'{args.model_training}_Autoencoder_MSE_Losses', args.output)

    classifiers, classifier_losses, classifier_accuracies = train_classifier(
        model=encoders,
        dataset=args.dataset,
        mode=args.classifier_training,
        epochs=args.classifier_epochs,
        batch_size=16,
        encoded_dim=encoded_dim,
        adj_matrix=A)
    
    plot_losses(classifier_losses, f'Autoencoder_{args.classifier_training}_Classifier_Losses', args.output)
    plot_accuracies(classifier_accuracies, f'Autoencoder_{args.classifier_training}_Classifier_Accuracies', args.output)

    test_accuracies, confusion_matrices = test_classifier(
        model=encoders,
        classifier=classifiers,
        dataset=args.dataset,
        mode=args.testing)
    
    for i, cm in enumerate(confusion_matrices):
        plot_confusion_matrix(cm, args.output, i)
    
    save_accuracies(test_accuracies, args.output)

def train_AE(mode: str, dataset: str, batch_size: int, epochs: int, encoded_dim: int, adj_matrix, lr: float=5e-3, device: str='cuda:0', n_workers: int=5):
    train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor()
        ])
    
    if dataset=='MNIST':
        channels = 1
        trainloaders = prepare_MNIST(mode, batch_size, train_transform)
    elif dataset=='CIFAR':
        channels = 3
        trainloaders = prepare_CIFAR(mode, batch_size, train_transform)

    worker_losses = {0: [], 1: [], 2: [], 3: [], 4: []}
    encoders = [Encoder(channels, encoded_dim).to(device) for _ in range(n_workers)]
    decoders = [Decoder(channels, encoded_dim).to(device) for _ in range(n_workers)]
    
    es = EarlyStopper()

    params_to_optimize = []
    for i in range(n_workers):
        params_to_optimize.append([
            {'params': encoders[i].parameters()},
            {'params': decoders[i].parameters()}
        ])

    criterion = nn.MSELoss()
    optimizers = [torch.optim.Adam(params, lr=lr) for params in params_to_optimize]
    #schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3) for optimizer in optimizers]

    for i in range(n_workers):
        encoders[i].train()
        decoders[i].train()

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
                #schedulers[k].step(loss)

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
    parser.add_argument('--topology', type=str, help='choose a network topology to organise worker nodes', default='random')

    args = parser.parse_args()

    main(args)