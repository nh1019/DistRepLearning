import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import numpy as np

from models.contrastive_learning import SimCLR, InfoNCELoss, TwoCropsTransform
from models.autoencoder import Encoder
from utils.earlystopping import EarlyStopper
from utils.prepare_dataloaders import prepare_MNIST, prepare_CIFAR
from utils.save_config import save_config
from utils.centralized_plotting import *
from classifiers.centralized_classifier import *

def main(args):
    save_config(args)

    encoders, losses = train_simCLR(
        mode=args.model_training,
        dataset=args.dataset,
        batch_size=256,
        epochs=args.model_epochs,
        optimizer=args.optimizer,
        warmup_epochs=args.warmup_epochs,
        scheduler=args.scheduler,
        encoded_dim=args.encoded_dim)
    
    plot_losses(losses, f'{args.model_training}_SimCLR_Losses', args.output)
    
    classifier, classifier_losses, classifier_accuracies = train_classifier(
        model=encoders,
        dataset=args.dataset,
        mode=args.classifier_training,
        epochs=args.classifier_epochs,
        batch_size=16,
        optimizer=args.optimizer,
        warmup_epochs=args.warmup_epochs,
        scheduler=args.scheduler,
        encoded_dim=args.encoded_dim)
    
    plot_losses(classifier_losses, f'{args.model_training}_SimCLR_{args.classifier_training}_Classifier_Losses', args.output)
    plot_accuracies(classifier_accuracies, f'{args.model_training}_SimCLR_{args.classifier_training}_Classifier_Accuracies', args.output)

    test_accuracies = test_classifier(
        model=encoders,
        classifier=classifier,
        dataset=args.dataset,
        mode=args.testing)
    
    save_accuracy(test_accuracies, args.output)

def train_simCLR(mode: str, 
                 dataset: str, 
                 epochs: int, 
                 batch_size: int,  
                 optimizer: str,
                 warmup_epochs: int,
                 scheduler: bool,
                 encoded_dim: int=128,
                 lr: float=1e-3, 
                 device: str='cuda:0'):
    
    train_transform = transforms.Compose([
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    es = EarlyStopper(min_delta=0.2)
    epoch_losses = []
    
    if dataset=='MNIST':
        trainloader = prepare_MNIST(mode, batch_size, TwoCropsTransform(train_transform))
    elif dataset=='CIFAR':
        channels = 3
        trainloader = prepare_CIFAR(mode, batch_size, TwoCropsTransform(train_transform))

    #encoder = Encoder(channels, encoded_dim).to(device)
    model = SimCLR().to(device)

    if warmup_epochs:
        desired_lr = lr
        initial_lr = desired_lr/100
    else:
        initial_lr = lr

    if optimizer=='Adam':
        optim = torch.optim.Adam(model.parameters(), lr=initial_lr)
    elif optimizer=='AdamW':
        optim = torch.optim.AdamW(model.parameters(), lr=initial_lr)
    elif optimizer=='SGD':
        optim = torch.optim.SGD(model.parameters(), lr=initial_lr)

    
    custom_loss = InfoNCELoss(device, batch_size).to(device)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(warmup_epochs):
        current_lr = initial_lr + (desired_lr-initial_lr)*(epoch/warmup_epochs)
        for param_group in optim.param_groups:
            param_group['lr'] = current_lr
        for batch_idx, (images, _) in tqdm(enumerate(trainloader)):
            images = torch.cat(images, dim=0)
            images = images.to(device)

            optim.zero_grad()
            features = model(images)
            logits, labels = custom_loss(features)

            loss = criterion(logits, labels)
            loss.backward()
            optim.step()

    if scheduler:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=5, factor=0.5, verbose=True)

    for epoch in range(epochs):
        curr_loss = []
        trainloader = trainloader
        for batch_idx, (images, _) in tqdm(enumerate(trainloader)):
            images = torch.cat(images, dim=0)
            images = images.to(device)

            optim.zero_grad()
            features = model(images)
            logits, labels = custom_loss(features)
            loss = criterion(logits, labels)
            curr_loss.append(loss.item())

            loss.backward()
            optim.step()

            if batch_idx%len(trainloader)==len(trainloader)-1:
                avg_train_loss = np.mean(curr_loss)
                print(f'In epoch {epoch}, average training loss is {avg_train_loss}.')
                epoch_losses.append(avg_train_loss)

        if scheduler:
            sched.step(avg_train_loss)

        if es.early_stop(avg_train_loss):
            print(f'Stopped training autoencoder after epoch {epoch}.')
            break

    return model, epoch_losses


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
