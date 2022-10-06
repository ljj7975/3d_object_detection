import vtk
from typing import List
from scene_vis.vtk_wrapper.vtk_utils import ToggleActorsInteractorStyle

class VtkRenderWindow(vtk.vtkRenderWindow):
    """Custom class based on VtkRenderWindow that handles many of the boilerplate configurations"""

    def __init__(self, window_name="roboeye", width=1920, height=1280):
        self.SetWindowName(
            window_name
        )
        self.SetSize(width, height)

        self.vtk_renderer = vtk.vtkRenderer()
        self.vtk_renderer.SetBackground(1.0, 1.0, 1.0)
        self.vtk_renderer.SetUseDepthPeeling(True)

        self.vtk_gt_collection = vtk.vtkPropCollection()

    def add_actor(self, vtk_actor:vtk.vtkActor):
        self.vtk_renderer.AddActor(vtk_actor)
        self.vtk_gt_collection.AddItem(vtk_actor)

    def add_actors(self, vtk_actors: List[vtk.vtkActor]):
        for actor in vtk_actors:
            self.add_actor(actor)

    def render(self):
        self.AddRenderer(self.vtk_renderer)

        vtk_render_window_interactor = vtk.vtkRenderWindowInteractor()
        vtk_render_window_interactor.SetRenderWindow(self)
        vtk_render_window_interactor.SetInteractorStyle(
            ToggleActorsInteractorStyle(
                [self.vtk_gt_collection],
                self.vtk_renderer,
            )
        )

        # Render in VTK
        self.Render()
        vtk_render_window_interactor.Start()  # Blocking
