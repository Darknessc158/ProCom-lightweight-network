import torch
from torch import nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    def __init__(self, alpha=0.5, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.alpha = alpha
        self.weight = weight
        self.size_average = size_average

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)

        binary_inputs = torch.argmax(inputs, keepdim = True)

        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = torch.mul(binary_inputs , targets).sum()                       
        dice_loss = 1 - (2.*intersection + smooth)/(binary_inputs.sum() + targets.sum() + smooth) 

        print(intersection)

        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        print(binary_inputs.sum())
        print(targets.sum())
        print("BCE")
        print(BCE)
        print(dice_loss)
        #Dice_BCE = self.alpha*BCE + self.alpha*dice_loss
        Dice_BCE = BCE
        
        return Dice_BCE
