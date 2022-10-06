import torch
import random
import numpy as np
from scipy.spatial.transform import Rotation


def sample_uniform(pc:torch.Tensor, num_points:int):
    return pc[random.sample(range(len(pc)), num_points)]

def pc_transforms(pc:torch.Tensor, normalize:bool = True, random_rotate:bool = True):
    if normalize:
        dx, dy, dz = np.ptp(pc, axis=0)
        diag_length = np.linalg.norm([dx, dy, dz], axis=0)
        pc = pc /diag_length

    if random_rotate:
        random_rotation = Rotation.random().as_matrix()
        pc = torch.matmul(pc, torch.Tensor(random_rotation))

    return pc
