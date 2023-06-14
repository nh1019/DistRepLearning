import torch

def calculate_l2_norm(model1, model2):
    l2_norm = 0.0
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        diff = param1 - param2
        l2_norm += torch.norm(diff, p=2)
    return l2_norm.item()

def calculate_mean_norm(models: list):
    pairwise_norms = []

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            l2_norm = calculate_l2_norm(models[i], models[j])
            pairwise_norms.append(l2_norm)

    return sum(pairwise_norms)/len(pairwise_norms)