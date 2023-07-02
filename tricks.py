import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    #  适用于样本不平衡的情况
    def __init__(self, weight=None, label_smoothing=0, reduction='mean', alpha=0.25, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction, label_smoothing=label_smoothing)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp   # alpha
        return loss.mean()


