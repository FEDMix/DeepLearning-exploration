from unittest import TestCase

import torch
import numpy as np
from losses import EdgeWeightedLoss, WeightedSoftDiceLoss, SoftDiceLoss

PRECISION = 3  # Because smoothing factors not super high precision


class TestEdgeWeightedLoss(TestCase):
    def test_all_ones_constant_weight(self):
        """ Edge Weighted Dice should be equal to normal dice if there are no boundaries."""
        target = torch.ones((1, 2, 2, 1))
        probs = torch.ones_like(target)
        result = self.loss.forward(probs, target).item()
        np.testing.assert_almost_equal(result, -1.0, decimal=PRECISION)

    def test_random_constant_weight_matches_dice(self):
        """ Edge Weighted Dice should be equal to normal dice if there are no boundaries."""
        dice = SoftDiceLoss()
        target = torch.from_numpy(np.random.rand(5, 5, 5, 5))
        probs = torch.from_numpy(np.random.rand(5, 5, 5, 5))
        result = self.loss.forward(probs, target).item()
        np.testing.assert_almost_equal(result, dice.forward(probs, target).item(), decimal=PRECISION)

    def test_greater_loss_near_boundary(self):
        target_array = np.zeros(1, 500, 500, 1)
        target_array[:, 100, :, :] = 1
        target = torch.from_numpy(target_array)

        probs = np.array(target_array)
        probs[:, 98, :, :] = 1
        probs_with_errors_near_edge = torch.from_numpy(probs)

        probs1 = np.array(target_array)
        probs1[:, 400, :, :] = 1
        probs_with_errors_far_from_edge = torch.from_numpy(probs1)

        errors_near_edge = self.loss.forward(probs_with_errors_near_edge, target).item()
        errors_far_from_edge = self.loss.forward(probs_with_errors_far_from_edge, target).item()

        self.assertGreater(errors_near_edge, errors_far_from_edge)

    def setUp(self):
        self.weighted_dice = WeightedSoftDiceLoss()
        self.loss = EdgeWeightedLoss(self.weighted_dice)
