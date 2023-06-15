import torch

def calculate_pairwise_l2_norm(model1, model2):
    l2_norm = 0.0
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        normalised_param1 = param1.flatten() / torch.norm(param1.flatten(), p=2)
        normalised_param2 = param2.flatten() / torch.norm(param2.flatten(), p=2)
        diff = normalised_param1 - normalised_param2
        l2_norm += torch.norm(diff, p=2)
    return l2_norm.item() / len(model1.parameters())

def calculate_mean_norm(models: list):
    pairwise_norms = []

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            l2_norm = calculate_pairwise_l2_norm(models[i], models[j])
            pairwise_norms.append(l2_norm)

    return sum(pairwise_norms)/len(pairwise_norms)