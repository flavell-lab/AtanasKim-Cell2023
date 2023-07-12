import torch
import torch.nn as nn
from .metrics import dice_coefficient, dice_coefficient_with_logits

class PixelwiseBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(PixelwiseBCEWithLogitsLoss, self).__init__()
        self.bce_logits = nn.BCEWithLogitsLoss(reduction="none")
        
    def forward(self, input, target, weights):
        mask = [weights != 0]
        result = (weights * self.bce_logits(input, target))[mask]
        
        return result.mean()

class PixelWiseCELogitsLoss(nn.Module):
    def __init__(self, weight_class):
        super(PixelWiseCELogitsLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight_class, reduction="none")
        
    def forward(self, input, target, weights):
        weights = weights.squeeze(1)
        target = target.squeeze(1)
        mask = [weights != 0]
        result = (weights * self.ce(input, target))[mask]

        return result.mean()
    
class DiceCoefficientLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(DiceCoefficientLoss, self).__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true):
        result = dice_coefficient(y_pred, y_true, self.epsilon)
        
        return result.mean()
        
class DiceCoefficientWithLogitsLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(DiceCoefficientWithLogitsLoss, self).__init__()
        self.epsilon = epsilon
        
    def forward(self, y, y_true):
        result = dice_coefficient_with_logits(y, y_true, self.epsilon)
        
        return result.mean()
