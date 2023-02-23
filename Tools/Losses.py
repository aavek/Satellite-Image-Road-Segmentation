import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#Segmentations
class mIoULoss(nn.Module):
    def __init__(self, weight=None, n_classes=2):
        super().__init__()
        self.classes = n_classes
        self.epsilon = 1e-8
        self.weights = weight * weight

    def forward(self, inputs, target):
        N = inputs.size()[0]
        ntar, htar, wtar = target.size()
        target_oneHot = torch.zeros(ntar, self.classes, htar, wtar, dtype = torch.long).cuda()
        target_oneHot = target_oneHot.scatter_(1, target.long().view(ntar, 1, htar, wtar), 1)
        inputs = F.softmax(inputs, dim=1)
        inter = inputs * target_oneHot
        inter = inter.view(N, self.classes, -1).sum(2)
        union = inputs + target_oneHot - (inputs * target_oneHot)
        union = union.view(N, self.classes, -1).sum(2)
        loss = (self.weights * inter) / (self.weights * union + self.epsilon)
        
        return -torch.mean(loss)

# Orientations # https://discuss.pytorch.org/t/nllloss-vs-crossentropyloss/92777 
class CrossEntropyLossImage(nn.Module): #https://discuss.pytorch.org/t/multi-class-cross-entropy-loss-function-implementation-in-pytorch/19077/12
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super().__init__()
        self.CE_loss = nn.CrossEntropyLoss(weight = weight, ignore_index = ignore_index, reduction = reduction)

    def forward(self, inputs, targets):
        return self.CE_loss(inputs, targets.long().cuda())