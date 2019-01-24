import torch
import torch.nn as nn
from torch.nn import Conv2d
import numpy as np
from scipy import stats as st

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, probs, targets):
        smooth = 1e-5

        m1 = probs.view(-1)
        m2 = targets.view(-1)

        intersection = torch.dot(m1, m2)
        union = torch.sum(m1) + torch.sum(m2)

        score = (2. * intersection + smooth) / (union + smooth)
        score = -score
        return score


class BCELoss2d(nn.Module):
    def __init__(self):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, probs, targets):
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs, targets_flat)


class WeightedBCELoss2d(nn.Module):
    def __init__(self):
        super(WeightedBCELoss2d, self).__init__()

    def forward(self, probs, targets, weights):
        w = weights.view(-1)
        probs = logits.view(-1)
        targets_flat = labels.view(-1)
        # http://geek.csdn.net/news/detail/126833
        loss = torch.log(probs) * targets_flat + \
               (torch.log(1 - probs)) * (1 - targets_flat)
        loss = loss * w
        loss = loss.sum() / w.sum()
        return loss


class CombinedLoss(nn.Module):
    def __init__(self, is_dice_log):
        super(CombinedLoss, self).__init__()
        self.is_dice_log = is_dice_log
        self.bce = BCELoss2d()
        self.soft_dice = SoftDiceLoss()

    def forward(self, probs, targets):

        bce_loss = self.bce(probs, targets)
        dice_loss = self.soft_dice(probs, targets)

        if self.is_dice_log:
            l = bce_loss - torch.log(-dice_loss)
        else:
            l = bce_loss + dice_loss

        return l, bce_loss, dice_loss


class WeightedSoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, probs, targets, weights):
        smooth = 1e-5

        m1 = probs.view(-1)
        m2 = targets.view(-1)
        w = weights.view(-1)
        w2 = w * w

        intersection = m1 * m2 * w2
        union = torch.sum(m1 * w2) + torch.sum(m2 * w2)

        score = (2. * intersection + smooth) / (union + smooth)
        score = -score
        return score


class WeightedSoftDiceLoss(nn.Module):
    def __init__(self):
        super(WeightedSoftDiceLoss, self).__init__()

    def forward(self, probs, targets, weights):
        smooth = 1e-5
        num = targets.size(0)
        w = weights.view(num, -1)
        w2 = w * w
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * ((w2 * intersection).sum(1) + smooth) / \
                ((w2 * m1).sum(1) + (w2 * m2).sum(1) + smooth)
        score = -score.sum() / num
        return score


class EdgeWeightedLoss(nn.Module):
    def __init__(self, weightedLoss):
        super(EdgeWeightedLoss, self).__init__()
        self.weightedLoss = weightedLoss
        self.conv = Conv2d(1, 1, 15, padding=7)
        array = [[_create_gaussian_kernel(15)]]
        kernel = torch.tensor(array, dtype=torch.float)
        self.conv.weight = torch.nn.Parameter(data=kernel)

    def forward(self, probs, targets):
        weights = torch.ones_like(probs)  # self._smoothed_edges(probs)
        return self.weightedLoss(probs, targets, weights)

    def _smoothed_edges(self, x):
        combined_edges = _get_combined_edges(x)
        print(x.shape)
        return self.conv.forward(combined_edges.view([1, 1, 512, 512]))


def _create_gaussian_kernel(kernlen=5, nsig=3):
    """Returns a 2D Gaussian kernel array. Taken from https://stackoverflow.com/a/29731818"""
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return np.array(kernel, dtype=np.float)


def _get_edges(x):
    xx = torch.zeros_like(x)
    xy = torch.zeros_like(x)
    xx[:, :-1, :] = x[:, 1:, :] - x[:, :-1, :]
    xy[:, :, :-1] = x[:, :, 1:] - x[:, :, :-1]
    return xx, xy


def _get_combined_edges(x):
    xx, xy = _get_edges(x)
    return torch.sqrt((xx * xx) + (xy * xy))


