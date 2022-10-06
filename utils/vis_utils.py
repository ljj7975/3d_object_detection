import torch

from scene_vis.vtk_wrapper.vtk_pc_actor import VtkPcActor
from scene_vis.vtk_wrapper.vtk_render_window import VtkRenderWindow
import numpy as np

def render_pc(pc):
    if isinstance(pc, torch.Tensor):
        pc = np.asarray(pc)
    vtk_render_window = VtkRenderWindow()
    vtk_scan_pc = VtkPcActor(pc, opacity=0.3, point_size=3)
    vtk_render_window.add_actors([vtk_scan_pc])
    vtk_render_window.render()