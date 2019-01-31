from unittest import TestCase

import torch
import numpy as np
from losses import EdgeWeightedLoss, WeightedSoftDiceLoss, SoftDiceLoss

PRECISION = 6  # Because smoothing factors not super high precision


class TestEdgeWeightedLoss(TestCase):
    def test_all_ones_constant_weight(self):
        """ Edge Weighted Dice should be equal to normal dice if there are no boundaries."""
        target = torch.ones((1, 1, 500, 500))
        probs = torch.ones_like(target)
        result = self.loss.forward(probs, target).item()
        np.testing.assert_almost_equal(result, -1.0, decimal=PRECISION)

    def test_greater_loss_near_boundary(self):
        target = self._create_even_tensor_with_single_edge_at(100)
        loss_near_edge = self._get_loss_with_ones_at_column(98, target)
        columns_far_from_edge = [5, 25, 45, 160, 250, 300, 450, 499]
        losses = [self._get_loss_with_ones_at_column(col, target) for col in columns_far_from_edge]
        self._assert_x_greater_than_all_in_y(loss_near_edge, losses)

    @staticmethod
    def _assert_x_greater_than_all_in_y(x_scalar, y_array):
        np.testing.assert_array_less(y_array, np.full_like(y_array, x_scalar))

    @staticmethod
    def _create_even_tensor_with_single_edge_at(column):
        target_array = np.zeros((1, 1, 500, 500))
        target_array[:, :, column, :] = 1
        target = torch.Tensor(target_array)
        return target

    def _get_loss_with_ones_at_column(self, i, target):
        probs = np.array(target.numpy())
        probs[:, :, i, :] = 1
        probs_with_errors_near_edge = torch.from_numpy(probs)
        errors_near_edge = self.loss.forward(probs_with_errors_near_edge, target).item()
        return errors_near_edge

    def setUp(self):
        self.weighted_dice = WeightedSoftDiceLoss()
        self.loss = EdgeWeightedLoss(self.weighted_dice)


class TestWeightedSoftDiceLoss(TestCase):
    def test_random_constant_weight_matches_dice(self):
        """ Edge Weighted Dice should be equal to normal dice if there are no boundaries."""
        target = torch.from_numpy(np.random.rand(5, 5, 500, 500))
        probs = torch.from_numpy(np.random.rand(5, 5, 500, 500))
        weights = torch.ones_like(target)
        result = WeightedSoftDiceLoss().forward(probs, target, weights).item()
        np.testing.assert_almost_equal(result, SoftDiceLoss()(probs, target).item(), decimal=PRECISION)
