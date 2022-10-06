from model_net_dataset import ModelNetDataset
from constants import DatasetSplit
import unittest
import pc_augmentation
from utils import vis_utils
import torch

class PointCloudAugmentationTest(unittest.TestCase):

    def setUp(self):
        val_set = ModelNetDataset(
            dataset_split=DatasetSplit.VAL
        )

        chair_sample_idx = 104
        self.pc, _ = val_set[chair_sample_idx]

    def test_sample_uniform(self):
        visualize = False

        num_pts = 1024
        sampled_pc = pc_augmentation.sample_uniform(self.pc, num_pts)
        self.assertEqual(len(sampled_pc), num_pts)
        if visualize:
            vis_utils.render_pc(sampled_pc)

        transformed_pc = pc_augmentation.pc_transforms(sampled_pc)
        if visualize:
            vis_utils.render_pc(transformed_pc)

        min_vals, _ = torch.min(transformed_pc, axis=0)
        max_vals, _ = torch.max(transformed_pc, axis=0)

        self.assertTrue((max_vals <= 1).all())
        self.assertTrue((min_vals >= -1).all())
