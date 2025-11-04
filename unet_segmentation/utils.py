import torch

def calculate_iou(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    target = target.float()

    intersection  = (pred * target).sum()
    union = (pred + target).clamp(0, 1).sum()

    if union == 0:
        return 1

    iou = intersection/union

    return iou.item()

def calculate_dice(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    target = target.float()

    intersection  = (pred * target).sum()
    
    pred_sum = pred.sum()
    target_sum = target.sum()

    dice = 2*intersection / (pred_sum+target_sum)

    return dice.item()