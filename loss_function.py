import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, activation='sigmoid'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.activation = activation
        
    def forward(self, pred, target):
        if self.activation == 'sigmoid':
            pred = torch.sigmoid(pred)
        
        # Flatten predictions and targets
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1.0 - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', activation='sigmoid'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.activation = activation
        
    def forward(self, pred, target):
        # Calculate probability and BCE based on activation type
        if self.activation == 'sigmoid':
            pred_prob = torch.sigmoid(pred)
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        else:
            pred_prob = pred  # Already probability values
            bce = F.binary_cross_entropy(pred, target, reduction='none')
        
        # Calculate focal weight
        pt = target * pred_prob + (1 - target) * (1 - pred_prob)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        
        # Calculate final loss
        loss = alpha_weight * focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0, activation='sigmoid'):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.activation = activation
        
    def forward(self, pred, target):
        if self.activation == 'sigmoid':
            pred = torch.sigmoid(pred)
        
        # Flatten predictions and targets
        pred = pred.view(-1)
        target = target.view(-1)
        
        # True Positive, False Positive, False Negative
        TP = (pred * target).sum()
        FP = (pred * (1 - target)).sum()
        FN = ((1 - pred) * target).sum()
        
        # Tversky index
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - tversky

class CombinedLoss(nn.Module):
    def __init__(self, config):
        super(CombinedLoss, self).__init__()
        # Load configuration
        self.dice_weight = config['loss']['dice_weight']
        self.bce_weight = config['loss']['bce_weight']
        self.focal_weight = config['loss']['focal_weight']
        
        # Get activation type, default to 'sigmoid' if not specified
        activation = config['loss'].get('activation', 'sigmoid')
        
        # Initialize loss components
        if activation == 'sigmoid':
            self.bce = nn.BCEWithLogitsLoss()
        else:
            self.bce = nn.BCELoss()
        
        self.dice = DiceLoss(activation=activation)
        self.focal = FocalLoss(
            alpha=config['loss']['focal_alpha'],
            gamma=config['loss']['focal_gamma'],
            activation=activation
        )
        self.tversky = TverskyLoss(
            alpha=config['loss']['tversky_alpha'],
            beta=config['loss']['tversky_beta'],
            activation=activation
        )
    
    def forward(self, pred, target):
        # Calculate component losses
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        focal_loss = self.focal(pred, target)
        tversky_loss = self.tversky(pred, target)
        
        # Combined loss
        loss = (self.bce_weight * bce_loss + 
                self.dice_weight * dice_loss + 
                self.focal_weight * focal_loss +
                (1 - self.bce_weight - self.dice_weight - self.focal_weight) * tversky_loss)
        
        return loss, {
            'bce': bce_loss.item(),
            'dice': dice_loss.item(),
            'focal': focal_loss.item(),
            'tversky': tversky_loss.item()
        }

def get_loss_function(config):
    """Get loss function based on configuration"""
    loss_type = config['loss']['type'].lower()
    activation = config['loss'].get('activation', 'sigmoid')
    
    if loss_type == 'bce':
        return nn.BCEWithLogitsLoss() if activation == 'sigmoid' else nn.BCELoss()
    elif loss_type == 'dice':
        return DiceLoss(activation=activation)
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=config['loss']['focal_alpha'],
            gamma=config['loss']['focal_gamma'],
            activation=activation
        )
    elif loss_type == 'tversky':
        return TverskyLoss(
            alpha=config['loss']['tversky_alpha'],
            beta=config['loss']['tversky_beta'],
            activation=activation
        )
    elif loss_type == 'combined':
        return CombinedLoss(config)
    else:
        raise ValueError(f"Unsupported loss function type: {loss_type}")