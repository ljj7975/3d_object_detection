from torch.utils.data import Dataset
import open3d as o3d
from typing import List
import os
import random
from constants import DatasetSplit
import torch
import numpy as np
from utils import vis_utils

class ModelNetDataset(Dataset):
    def __init__(
        self,
        data_dir:str = "ModelNet10",
        pos_objects:List[str] = ["bed", "chair", "desk", "monitor", "table"],
        neg_objects:List[str] = None,
        dataset_split:DatasetSplit = DatasetSplit.TRAIN,
        train_set_ratio: float = 0.8,
        neg_sample_ratio:float = 0.3 # % of negative samples in the dataset
    ):
        self.dataset_split = dataset_split
        self.data_dir = data_dir
        self.dir_list = data_dir
        self.pos_objects = pos_objects
        self.neg_objects = neg_objects
        if neg_objects is None:
            self.neg_objects = []
            for dir_name in os.listdir(self.data_dir):
                if os.path.isdir(os.path.join(self.data_dir, dir_name)) and dir_name not in self.pos_objects:
                    self.neg_objects.append(dir_name)
        self.neg_label = len(self.pos_objects)

        self.train_set_ratio = train_set_ratio
        self.sample_paths = []
        self.labels = []
        self.neg_sample_ratio = neg_sample_ratio
        self.counts_per_label = []

        self._preload_samples()

        # for now unlimited
        self.cache = {}

    def _split_train_val(self, sample_paths:List[str]):
        if self.dataset_split == DatasetSplit.TRAIN:
            sample_paths = sample_paths[:int(len(sample_paths) * self.train_set_ratio)]
        elif self.dataset_split == DatasetSplit.VAL:
            sample_paths = sample_paths[int(len(sample_paths) * self.train_set_ratio):]
        return sample_paths

    def _preload_samples(self):

        split_dir_name = self.dataset_split.value
        if self.dataset_split == DatasetSplit.VAL:
            split_dir_name = DatasetSplit.TRAIN.value

        for obj_idx, object in enumerate(self.pos_objects):
            obj_sample_paths = []
            for file_name in os.listdir(os.path.join(self.data_dir, object, split_dir_name)):
                obj_sample_paths.append(
                    os.path.join(self.data_dir, object, split_dir_name, file_name)
                )

            obj_sample_paths = self._split_train_val(obj_sample_paths)

            self.sample_paths.extend(obj_sample_paths)
            self.labels.extend([obj_idx] * len(obj_sample_paths))
            self.counts_per_label.append(len(obj_sample_paths))

        neg_sample_paths = []
        for object in self.neg_objects:
            neg_obj_sample_paths = []
            for file_name in os.listdir(os.path.join(self.data_dir, object, split_dir_name)):
                neg_obj_sample_paths.append(
                    os.path.join(self.data_dir, object, split_dir_name, file_name)
                )

            neg_obj_sample_paths = self._split_train_val(neg_obj_sample_paths)

            neg_sample_paths.extend(neg_obj_sample_paths)

        random.shuffle(neg_sample_paths)
        max_num_neg_samples = int(len(self.sample_paths) / (1-self.neg_sample_ratio) * self.neg_sample_ratio)
        num_neg_samples = min(max_num_neg_samples, len(neg_sample_paths))
        self.sample_paths.extend(neg_sample_paths[:num_neg_samples])
        self.labels.extend([self.neg_label] * num_neg_samples)
        self.counts_per_label.append(num_neg_samples)

        total_count = 0
        print("counts per label")
        for label, count in enumerate(self.counts_per_label):
            print(f"\t{label}: {count}")
            total_count += count

        assert len(self.labels) == len(self.sample_paths)
        assert len(self.labels) == total_count

    def _load_pc(self, file_name):
        if file_name not in self.cache:
            o3d_mesh = o3d.io.read_triangle_mesh(file_name)
            self.cache[file_name] = np.asarray(o3d_mesh.vertices)
        return self.cache[file_name]

    def visualize_sample(self, idx):
        file_name = self.sample_paths[idx]
        print(f"Visualizing sample {file_name}")
        vis_utils.render_pc(self._load_pc(file_name))

    def __len__(self):
        """return number of samples"""
        return len(self.sample_paths)

    def __getitem__(self, idx):
        """loads and returns a sample from the dataset at the given index"""
        np_pc = self._load_pc(self.sample_paths[idx])
        return torch.Tensor(np_pc), self.labels[idx]