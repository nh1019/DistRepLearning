import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset


def prepare_MNIST(mode: str, batch_size: int, train_transform=None, train=True, data_fraction=1.):
    test_transform = transforms.Compose([transforms.ToTensor()])
    worker_classes = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    if train:
        train_dataset = datasets.MNIST(root='./data', train=True, transform=train_transform, download=True)

        if mode=='centralized':
            if data_fraction<1.:
                num_samples = int(data_fraction*len(train_dataset))
                train_dataset = torch.utils.data.random_split(train_dataset, [num_samples, len(train_dataset)-num_samples])[0]

            return DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
        else:
            #separate data among workers by class
            worker_datasets = []
            for classes in worker_classes:
                idx = torch.cat([torch.where(train_dataset.targets==c)[0] for c in classes])
                worker_dataset = Subset(train_dataset, idx)
                if data_fraction<1.:
                        worker_samples = int(data_fraction*len(worker_dataset))
                        worker_dataset = torch.utils.data.random_split(worker_dataset, [worker_samples, len(worker_dataset)-worker_samples])[0]

                worker_datasets.append(worker_dataset)

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
        
def prepare_CIFAR(mode: str, batch_size: int, data_fraction: float=1., train_transform=None, train=True, iid=False):
    test_transform = transforms.Compose([transforms.ToTensor()])
    worker_classes = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    if train:
        train_dataset = datasets.CIFAR10(root='./data', train=True, transform=train_transform, download=True)

        if iid:
            if data_fraction<1.:
                num_samples = int(data_fraction*len(train_dataset))
                train_dataset = torch.utils.data.random_split(train_dataset, [num_samples, len(train_dataset)-num_samples])[0]
                
            l = int(len(train_dataset)/5)
            subsets = [Subset(train_dataset, range(i*l,(i+1)*l)) for i in range(0,5)]
            return [DataLoader(subset, batch_size, shuffle=True, drop_last=True) for subset in subsets]

        else:
            if mode=='centralized':
                if data_fraction<1.:
                    num_samples = int(data_fraction*len(train_dataset))
                    train_dataset = torch.utils.data.random_split(train_dataset, [num_samples, len(train_dataset)-num_samples])[0]

                return DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
            else:
                #separate data among workers by class
                worker_datasets = []
                for classes in worker_classes:
                    #need to convert list of targets to tensor
                    targets_tensor = torch.tensor(train_dataset.targets)
                    idx = torch.cat([torch.where(targets_tensor==c)[0] for c in classes])
                    worker_dataset = Subset(train_dataset, idx)
                    if data_fraction<1.:
                        worker_samples = int(data_fraction*len(worker_dataset))
                        worker_dataset = torch.utils.data.random_split(worker_dataset, [worker_samples, len(worker_dataset)-worker_samples])[0]
                    
                    worker_datasets.append(worker_dataset)

                #change batch size in case worker dataset is smaller than existing batch size
                dataset_lens = [len(x) for x in worker_datasets]
                batch_size = min(min(dataset_lens), batch_size)

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