from model_net_dataset import ModelNetDataset
from constants import DatasetSplit
import unittest
from utils import vis_utils

class ModelNetDatasetTest(unittest.TestCase):

    def test_dataset_creation(self):
        train_set = ModelNetDataset(
            dataset_split = DatasetSplit.TRAIN
        )

        self.assertEqual(len(train_set), 2812)

        val_set = ModelNetDataset(
            dataset_split = DatasetSplit.VAL
        )

        self.assertEqual(len(val_set), 705)

        test_set = ModelNetDataset(
            dataset_split = DatasetSplit.TEST
        )

        self.assertEqual(len(test_set), 697)

    def test_load_sample(self):
        val_set = ModelNetDataset(
            dataset_split = DatasetSplit.VAL
        )
        visualize = True

        bed_sample_idx = 0
        bed_pc, bed_label = val_set[bed_sample_idx]
        self.assertEqual(bed_label, 0)
        if visualize:
            val_set.visualize_sample(bed_sample_idx)

        chair_sample_idx = 104
        chair_pc, chair_label = val_set[chair_sample_idx]
        self.assertEqual(chair_label, 1)
        if visualize:
            val_set.visualize_sample(chair_sample_idx)

        desk_sample_idx = 282
        desk_pc, desk_label = val_set[desk_sample_idx]
        self.assertEqual(desk_label, 2)
        if visualize:
            val_set.visualize_sample(desk_sample_idx)

        monitor_sample_idx = 322
        monitor_pc, monitor_label = val_set[monitor_sample_idx]
        self.assertEqual(monitor_label, 3)
        if visualize:
            val_set.visualize_sample(monitor_sample_idx)

        table_sample_idx = 415
        table_pc, table_label = val_set[table_sample_idx]
        self.assertEqual(table_label, 4)
        if visualize:
            val_set.visualize_sample(table_sample_idx)

        neg_sample_idx = 494
        neg_pc, neg_label = val_set[neg_sample_idx]
        self.assertEqual(neg_label, 5)
        if visualize:
            val_set.visualize_sample(neg_sample_idx)

