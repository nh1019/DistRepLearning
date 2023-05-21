import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoCropsTransform:
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class SimSiam(nn.Module):
    def __init__(self, encoder, dim=128, pred_dim=32):
        super(SimSiam, self).__init__()

        self.encoder = encoder

        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, dim)
        )

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return p1, p2, z1.detach(), z2.detach()
    
class SimCLR(nn.Module):
    def __init__(self, encoder, channels, out_dim=32):
        super(SimCLR, self).__init__()

        self.encoder = encoder

        #mlp projection head
        self.dim_mlp = 7*7*32 if channels==1 else 8*8*32
        self.encoder.encoder_lin = nn.Sequential(
            nn.Linear(self.dim_mlp, self.dim_mlp), 
            nn.ReLU(),
            self.encoder.encoder_lin 
        )

    def forward(self, x):
        return self.encoder(x)

#custom loss for SimCLR
class InfoNCELoss(nn.Module):
    def __init__(self, device, batch_size, sim_fn='dot', n_views=2, temperature=0.7):
        super(InfoNCELoss, self).__init__()

        self.device = device
        self.batch_size = batch_size
        self.n_views = n_views
        self.temperature = temperature
        self.sim_fn = sim_fn

    def forward(self, features):
        labels = torch.cat([torch.arange(self.batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0)==labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        #similarity matrix
        if self.sim_fn=='dot':
            sim_matrix = torch.matmul(features, features.T)
        elif self.sim_fn=='cosine':
            sim = nn.CosineSimilarity(dim=-1)
            sim_matrix = sim(features.unsqueeze(1), features.unsqueeze(0))

        #discard the main diagonal from label and similarity matrices
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        sim_matrix = sim_matrix[~mask].view(sim_matrix.shape[0], -1)

        pos = sim_matrix[labels.bool()].view(labels.shape[0], -1)

        neg = sim_matrix[~labels.bool()].view(sim_matrix.shape[0], -1)

        logits = torch.cat([pos, neg], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits /= self.temperature

        return logits, labels




    
        

