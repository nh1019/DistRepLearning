import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import numpy as np

from models.contrastive_learning import SimCLR, InfoNCELoss, TwoCropsTransform
from models.autoencoder import Encoder
from utils.earlystopping import EarlyStopper
from utils.aggregate import aggregate, generate_graph
from utils.prepare_dataloaders import prepare_MNIST, prepare_CIFAR
from utils.save_config import save_config
from utils.dist_plotting import *
from scripts.dist_classifier import *

def main(args):
    save_config(args)

    A = generate_graph(5, args.topology)

    encoders, losses = train_simCLR(
        mode=args.model_training,
        dataset=args.dataset,
        batch_size=256,
        epochs=args.model_epochs,
        encoded_dim=args.encoded_dim,
        adj_matrix=A)
    
    plot_losses(losses, f'{args.model_training}_SimCLR_Losses', args.output)
    
    classifiers, classifier_losses, classifier_accuracies = train_classifier(
        model=encoders,
        dataset=args.dataset,
        mode=args.classifier_training,
        epochs=args.classifier_epochs,
        batch_size=16,
        encoded_dim=args.encoded_dim,
        adj_matrix=A)
    
    plot_losses(classifier_losses, f'{args.model_training}_SimCLR_{args.classifier_training}_Classifier_Losses', args.output)
    plot_accuracies(classifier_accuracies, f'{args.model_training}_SimCLR_{args.classifier_training}_Classifier_Accuracies', args.output)

    test_accuracies = test_classifier(
        model=encoders,
        classifier=classifiers,
        dataset=args.dataset,
        mode=args.testing)
    
    save_accuracies(test_accuracies, args.output)

def train_simCLR(mode: str, dataset: str, epochs: int, batch_size: int, adj_matrix, encoded_dim: int=128, lr: float=1e-3, device: str='cuda:0', n_workers: int=5):
    train_transform = transforms.Compose([
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    es = EarlyStopper(min_delta=0.5)
    worker_losses = {0: [], 1: [], 2: [], 3: [], 4: []}
    
    if dataset=='MNIST':
        channels = 1
        trainloaders = prepare_MNIST(mode, batch_size, TwoCropsTransform(train_transform))
    elif dataset=='CIFAR':
        channels = 3
        trainloaders = prepare_CIFAR(mode, batch_size, TwoCropsTransform(train_transform))

    encoders = [Encoder(channels, encoded_dim).to(device) for _ in range(n_workers)]
    models = [SimCLR(encoder, channels).to(device) for encoder in encoders]
    optimizers = [torch.optim.Adam(model.parameters(), lr=lr) for model in models]
    custom_loss = InfoNCELoss(device, batch_size)
    criterion = nn.CrossEntropyLoss()

    for model in models:
        model.train()

    for epoch in range(epochs):
        for k in range(n_workers):
            curr_loss = []
            trainloader = trainloaders[k]
            for batch_idx, (images, _) in tqdm(enumerate(trainloader)):
                images = torch.cat(images, dim=0)
                images = images.to(device)

                features = models[k](images)
                logits, labels = custom_loss(features)
                loss = criterion(logits, labels)

                optimizers[k].zero_grad()
                loss.backward()
                optimizers[k].step()

                if batch_idx%len(trainloader)==len(trainloader)-1:
                    avg_train_loss = np.mean(curr_loss)
                    print(f'In epoch {epoch} for worker {k}, average training loss is {avg_train_loss}.')
                    worker_losses[k].append(avg_train_loss)

        curr_average = np.mean([worker_losses[k][epoch] for k in worker_losses.keys()])
        if es.early_stop(curr_average):
            print(f'Stopped training autoencoder after epoch {epoch}.')
            break
                
        if mode=='collaborative':
            encoders = aggregate(n_workers, encoders, adj_matrix)

    return encoders, worker_losses


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
