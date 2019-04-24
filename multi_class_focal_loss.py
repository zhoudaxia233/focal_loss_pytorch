import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2, size_average=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets):
        targets = targets.view(-1, 1)
        
        prob_dist = F.softmax(inputs, dim=1)
        pt = prob_dist.gather(1, targets)
        
        batch_loss = -self.alpha * (torch.pow((1 - pt), self.gamma)) * torch.log(pt)
        
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
