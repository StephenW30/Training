import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score

def dice_coefficient(y_pred, y_true, smooth=1e-7):
    """Calculate Dice coefficient"""
    # Ensure binary values
    y_pred = (y_pred > 0.5).float()
    y_true = (y_true > 0.5).float()
    
    # Flatten data
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    
    # Calculate intersection and union
    intersection = (y_pred * y_true).sum()
    return (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)

def iou_score(y_pred, y_true, smooth=1e-7):
    """Calculate IoU/Jaccard index"""
    # Ensure binary values
    y_pred = (y_pred > 0.5).float()
    y_true = (y_true > 0.5).float()
    
    # Flatten data
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    
    # Calculate intersection and union
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def precision(y_pred, y_true):
    """Calculate precision"""
    y_pred = (y_pred > 0.5).float().cpu().numpy().flatten()
    y_true = y_true.cpu().numpy().flatten()
    return precision_score(y_true, y_pred, zero_division=1)

def recall(y_pred, y_true):
    """Calculate recall"""
    y_pred = (y_pred > 0.5).float().cpu().numpy().flatten()
    y_true = y_true.cpu().numpy().flatten()
    return recall_score(y_true, y_pred, zero_division=1)

def accuracy(y_pred, y_true):
    """Calculate accuracy"""
    y_pred = (y_pred > 0.5).float().cpu().numpy().flatten()
    y_true = y_true.cpu().numpy().flatten()
    return accuracy_score(y_true, y_pred)

def calculate_metrics(y_pred, y_true, metrics_list):
    """Calculate multiple evaluation metrics"""
    results = {}
    
    for metric in metrics_list:
        if metric == 'dice':
            results['dice'] = dice_coefficient(y_pred, y_true).item()
        elif metric == 'iou':
            results['iou'] = iou_score(y_pred, y_true).item()
        elif metric == 'precision':
            results['precision'] = precision(y_pred, y_true)
        elif metric == 'recall':
            results['recall'] = recall(y_pred, y_true)
        elif metric == 'accuracy':
            results['accuracy'] = accuracy(y_pred, y_true)