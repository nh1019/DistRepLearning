import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision.transforms import transforms

from models.autoencoder import Encoder, Decoder
from utils.prepare_dataloaders import prepare_MNIST, prepare_CIFAR
from utils.aggregate import aggregate
from utils.earlystopping import EarlyStopper


def train_AE(mode: str, dataset: str, batch_size: int, epochs: int, encoded_dim: int, adj_matrix, lr: float=1e-4, device: str='cuda:0', n_workers: int=5):
    train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor()
        ])
    
    if dataset=='MNIST':
        channels = 1
        trainloaders = prepare_MNIST(mode, batch_size, train_transform)
    elif dataset=='CIFAR':
        channels = 2
        trainloaders = prepare_CIFAR(mode, batch_size, train_transform)

    worker_losses = {0: [], 1: [], 2: [], 3: [], 4: []}
    encoders = [Encoder(channels, encoded_dim).to(device) for k in range(n_workers)]
    decoders = [Decoder(channels, encoded_dim).to(device) for k in range(n_workers)]
    
    es = EarlyStopper(min_delta=0.1)

    params_to_optimize = []
    for i in range(n_workers):
        params_to_optimize.append([
            {'params': encoders[i].parameters()},
            {'params': decoders[i].parameters()}
        ])

    criterion = nn.MSELoss()
    optimizers = [torch.optim.Adam(params, lr=lr) for params in params_to_optimize]
    schedulers = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9) for optimizer in optimizers]

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
        if mode=='collaborative':
            encoders = aggregate(n_workers, encoders, adj_matrix)
            decoders = aggregate(n_workers, decoders, adj_matrix)

    return encoders, worker_losses, encoded_dim