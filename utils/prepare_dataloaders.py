import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset


def prepare_MNIST(mode: str, batch_size: int, train_transform=None, train=True):
    test_transform = transforms.Compose([transforms.ToTensor()])
    worker_classes = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    if train:
        train_dataset = datasets.MNIST(root='./data', train=True, transform=train_transform, download=True)

        if mode=='centralized':
            return DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
        else:
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
        
def prepare_CIFAR(mode: str, batch_size: int, train_transform=None, train=True):
    test_transform = transforms.Compose([transforms.ToTensor()])
    #worker_classes = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]]
    worker_classes = [[3, 5], [5, 7], [6, 2], [0, 8], [1, 9]]

    if train:
        train_dataset = datasets.CIFAR10(root='./data', train=True, transform=train_transform, download=True)

        if mode=='centralized':
            return DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
        else:
            #separate data among workers by class
            worker_datasets = []
            for classes in worker_classes:
                #need to convert list of targets to tensor
                targets_tensor = torch.tensor(train_dataset.targets)
                idx = torch.cat([torch.where(targets_tensor==c)[0] for c in classes])
                worker_datasets.append(Subset(train_dataset, idx))

            trainloaders = []
            for i in range(len(worker_datasets)):
                trainloaders.append(DataLoader(worker_datasets[i], batch_size, shuffle=True, drop_last=True))

            return trainloaders
    
    else:
        test_dataset = datasets.CIFAR10(root='./data', train=False, transform=test_transform, download=True)

        if mode=='local':
            test_datasets = []
            for classes in worker_classes:
                targets_tensor = torch.tensor(test_dataset.targets)
                idx = torch.cat([torch.where(targets_tensor==c)[0] for c in classes])
                test_datasets.append(Subset(test_dataset, idx))
            
            testloaders = []
            for test_dataset in test_datasets:
                testloaders.append(DataLoader(test_dataset, batch_size, drop_last=True))

            return testloaders, test_datasets

        else:
            return DataLoader(test_dataset, batch_size, drop_last=True)