import torch
import matplotlib.pyplot as plt
import numpy as np


def DiceCoeff(prediction, target):
    smooth = 1e-5

    target = (target > 0.5).float()
    prediction = (prediction > 0.5).float()

    inter = torch.dot(prediction.view(-1), target.view(-1))
    union_sum = torch.sum(prediction) + torch.sum(target)
    union = union_sum + smooth
    t = (2 * inter.float() + smooth) / union.float()

    if union_sum.item() == 0:
        return 1

    return t.item()


def DiceCoeffBatch(input, target):
    """Dice coeff for batches"""
    s = 0.0

    for i in range(input.shape[0]):
        s = s + dice_coeff(input[i], target[i])

    s /= (i + 1)
    return s
