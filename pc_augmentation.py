import torch
import random
import numpy as np
from scipy.spatial.transform import Rotation


def pc_transforms(pc:torch.Tensor, normalize:bool = True, random_rotate:bool = True):
    """This function handles normalization and random rotation (SubTask 2)
    but it got split into two different functions (normalize, rotate_randomly) for usability
    """
    if normalize:
        dx, dy, dz = np.ptp(pc, axis=0)
        diag_length = np.linalg.norm([dx, dy, dz], axis=0)
        pc = pc /diag_length

    if random_rotate:
        random_rotation = Rotation.random().as_matrix()
        pc = torch.matmul(pc, torch.Tensor(random_rotation))

    return pc

def sample_uniform(pc:torch.Tensor, num_points:int):
    """Uniformly sample num_points data points from the point cloud"""
    num_valid_points = len(pc)
    if num_points < num_valid_points:
        pc = pc[random.sample(range(num_valid_points), num_points)]
    return pc

def pad_zeros(pc:torch.Tensor, num_points:int):
    """If there are insufficient points, pad them with zeros"""
    if len(pc) < num_points:
        pc = torch.zeros(num_points, 3)
        pc[:num_points, :] = pc
        pc = pc[torch.randperm(num_points)]
    return pc

def normalize(pc:torch.Tensor):
    """Normalize the point cloud data using the object dimension
    points lie between -1 and 1 after normalization"""
    dx, dy, dz = np.ptp(pc, axis=0)
    diag_length = np.linalg.norm([dx, dy, dz], axis=0)
    return pc / diag_length

def rotate_randomly(pc:torch.Tensor):
    """Randomly rotate the give point cloud"""
    random_rotation = Rotation.random().as_matrix()
    return torch.matmul(pc, torch.Tensor(random_rotation))

def translate_randomly(pc:torch.Tensor):
    """Randomly translate the give point cloud"""
    random_translation = (np.random.random(3) - 0.5) * 0.3
    return pc + random_translation

def add_gaussian_noise(pc:torch.Tensor):
    """Add small gaussian noise to each point"""
    return pc + 0.02 * torch.randn(pc.shape)
