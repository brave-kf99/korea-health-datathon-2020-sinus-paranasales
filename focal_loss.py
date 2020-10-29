import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossBinary(nn.Module):
    def __init__(self,alpha=1,gamma=2):
        super(FocalLossBinary,self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self,preds,truth):
        criterion = nn.BCEWithLogitsLoss()
        logits = criterion(preds,truth.unsqueeze(-1).type_as(preds))
        pt = torch.exp(-logits)
        focal_loss = self.alpha*(1-pt)**self.gamma*logits
        
        return torch.mean(focal_loss)

class FocalLoss(nn.Module):
    def __init__(self,device,alpha=5,gamma=2):
        super(FocalLoss,self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        
    def forward(self,preds,truth):
        logits = F.cross_entropy(preds,truth, reduction='none') # important to add reduction='none' to keep per-batch-item loss
        pt = torch.exp(-logits)
        focal_loss = self.alpha*(1-pt)**self.gamma*logits
        
        return torch.mean(focal_loss)