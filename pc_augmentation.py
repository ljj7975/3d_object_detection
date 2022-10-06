import torch
import random
import numpy as np
from scipy.spatial.transform import Rotation


def sample_uniform(pc:torch.Tensor, num_points:int):
    num_valid_points = len(pc)
    if num_points < num_valid_points:
        pc = pc[random.sample(range(num_valid_points), num_points)]
    return pc

def sample_uniform(pc:torch.Tensor, num_points:int):
    num_valid_points = len(pc)
    if num_points < num_valid_points:
        pc = pc[random.sample(range(num_valid_points), num_points)]
    return pc

def pad_zeros(pc, num_points):
    if len(pc) < num_points:
        pc = torch.zeros(num_points, 3)
        pc[:num_points, :] = pc
        pc = pc[torch.randperm(num_points)]
    return pc


def pc_transforms(pc:torch.Tensor, normalize:bool = True, random_rotate:bool = True):
    if normalize:
        dx, dy, dz = np.ptp(pc, axis=0)
        diag_length = np.linalg.norm([dx, dy, dz], axis=0)
        pc = pc /diag_length

    if random_rotate:
        random_rotation = Rotation.random().as_matrix()
        pc = torch.matmul(pc, torch.Tensor(random_rotation))

    return pc

def normalize(pc:torch.Tensor):
    dx, dy, dz = np.ptp(pc, axis=0)
    diag_length = np.linalg.norm([dx, dy, dz], axis=0)
    return pc /diag_length

def rotate_randomly(pc:torch.Tensor):
    random_rotation = Rotation.random().as_matrix()
    return torch.matmul(pc, torch.Tensor(random_rotation))

def add_gaussian_noise(pc:torch.Tensor):
    return pc + 0.05 * torch.randn(pc.shape)