"""A5消融目标的接口和掩码回归测试。"""
import unittest
import torch
from map_net_a5.metrics import masked_error_sums, water_valid_mask
from train import TRAIN_OBJECTIVE



class FullGridObjectiveTests(unittest.TestCase):
    def test_a5_training_counts_seabed_points(self):
        truth = torch.full((1, 9, 200, 250), 2.0)
        pred = truth.clone()
        pred[:, :, 100:, :] += 3.0
        bottom = torch.full((121, 9, 250), 49, dtype=torch.int16)
        water = water_valid_mask(truth, torch.tensor([0]), bottom)
        sq_water, _, n_water = masked_error_sums(pred, truth, water)
        sq_all, _, n_all = masked_error_sums(pred, truth, torch.ones_like(truth, dtype=torch.bool))
        assert sq_water.item() == 0
        assert sq_all.item() > 0
        assert int(n_all) == 9*200*250
        assert int(n_water) == 9*50*250

    def test_a5_objective_identity(self):
        assert TRAIN_OBJECTIVE == "A5_full_grid_train_mse_masked_validation"
