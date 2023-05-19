import torch

from utils.prepare_dataloaders import prepare_MNIST, prepare_CIFAR

def test_classifier(model, classifier, dataset: str, mode: str, device: str='cuda:0', n_workers: int=5):
    classifiers = classifier
    encoders = model
    
    for classifier in classifiers:
        classifier.eval()

    if dataset=='MNIST':
        testloaders = prepare_MNIST(mode, batch_size=8, train=False)
    elif dataset=='CIFAR':
        testloaders = prepare_CIFAR(mode, batch_size=8, train=False)

    worker_accuracies = {0: [], 1: [], 2: [], 3: [], 4: []}

    for k in range(n_workers):
        total = 0
        correct = 0
        testloader = testloaders[k]
        with torch.no_grad():
            for (features, labels) in testloader:
                features, labels = features.to(device), labels.to(device)
                reps = encoders[k](features)
                outputs = classifiers[k](reps)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted==labels).sum().item()
    
        worker_accuracies[k] = (correct/total)*100
    
    return worker_accuracies