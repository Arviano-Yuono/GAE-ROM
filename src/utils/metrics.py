
import torch

def mse_error(pred, target):
    assert pred.shape == target.shape, "Prediction and target must have the same shape"
    return torch.mean((pred - target) ** 2)

def relative_error(pred, target, epsilon = 1e-4):
    assert pred.shape == target.shape, "Prediction and target must have the same shape"
    return torch.sum((pred - target)**2) / (torch.sum(target**2))

def mae_error(pred, target):
    assert pred.shape == target.shape, "Prediction and target must have the same shape"
    return torch.mean(torch.abs(pred - target))

def rmse_error(pred, target):
    assert pred.shape == target.shape, "Prediction and target must have the same shape"
    return torch.sqrt(torch.mean((pred - target) ** 2))

def r2_error(pred, target):
    assert pred.shape == target.shape, "Prediction and target must have the same shape"
    return 1 - torch.sum((pred - target) ** 2) / torch.sum((target - torch.mean(target)) ** 2)

def r2_error(pred, target):
    assert pred.shape == target.shape, "Prediction and target must have the same shape"
    return 0