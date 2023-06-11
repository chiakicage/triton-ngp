import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import nerfacc

class Trainer:
    def __init__(self, network: nn.Module, cfg):
        self.network = network
        self.network = self.network.cuda()
        self.estimator = nerfacc.OccGridEstimator(
            roi_aabb=torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5]).cuda(),
        )
        self.optimizer = Adam(
            self.network.parameters(), lr=1e-2, weight_decay=1e-6
        )
    
    def train(self, data_loader):
        self.network.train()
        
        for i, data in tqdm(enumerate(data_loader)):
            # data = { k: v.cuda() for k, v in data.items() }
            rays_o = torch.tensor(data['rays_o']).cuda()
            rays_d = torch.tensor(data['rays_d']).cuda()
            rgb_gt = data['rgb']
            # uv = data['uv']

            rgb, density, _, _ = self.network(rays_o, rays_d, self.estimator)
            self.optimizer.zero_grad()
            loss = F.mse_loss(rgb, rgb_gt)
            loss.backward()
            self.optimizer.step()
            print(f'loss: {loss.item()}')

            


