from typing import Optional, Union, List

import numpy as np
import vtk
from scene_vis.vtk_wrapper import vtk_utils
from scene_vis.vtk_wrapper.vtk_lines_actor import VtkLinesActor
from scene_vis.vtk_wrapper.vtk_pc_actor import VtkPcActor
from scene_vis.vtk_wrapper.vtk_ply_actor import VtkPlyActor
from scene_vis.vtk_wrapper.vtk_text_labels import VtkTextLabels
from scene_vis.vtk_wrapper.vtk_utils import ToggleActorsInteractorStyle

from nova.core.mesh import Mesh


class VtkPlotter:
    def __init__(self, window_title, window_size=(960, 600), use_depth_peeling=False):
        """VTK plotter

        Args:
            window_title: Window title (required)
            window_size: Window width and height
            use_depth_peeling: Set to True to correctly render transparent objects
        """
        self.renderer, self.render_window = vtk_utils.setup_renderer_and_window(
            window_title=window_title, window_size=window_size
        )

        self.camera = self.renderer.GetActiveCamera()
        self.renderer.ResetCamera()

        self.interactor_style = ToggleActorsInteractorStyle([], self.renderer, self.camera)
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        self.interactor.SetInteractorStyle(self.interactor_style)

        # Reference to orientation widget
        self.orientation_widget = None

        if use_depth_peeling:
            self.renderer.SetUseDepthPeeling(True)

    def close(self):
        render_window = self.interactor.GetRenderWindow()
        render_window.Finalize()
        self.renderer.RemoveAllViewProps()
        self.interactor.TerminateApp()
        del self.render_window, self.interactor

    def add_actor(self, actor):
        """Add any actor to the renderer"""
        self.renderer.AddActor(actor)
        return actor

    def add_axes_actor(self, *, length=None):
        """Add axes actor to the scene.

        Notes:
            Avoid using a transform as only positive axes will be visible. Use add_axes
            if axes in negative positions are required.

        Args:
            length: Length for axes
        """
        axes_actor = vtk.vtkAxesActor()

        if length is not None:
            axes_actor.SetTotalLength(length, length, length)

        self.renderer.AddActor(axes_actor)

    def add_axes_widget(self):
        # Orientation widget, must keep a reference to it or will be automatically deleted
        if self.orientation_widget is None:
            axes = vtk.vtkAxesActor()
            self.orientation_widget = vtk.vtkOrientationMarkerWidget()
            self.orientation_widget.SetOrientationMarker(axes)
            self.orientation_widget.SetInteractor(self.interactor)
            self.orientation_widget.SetViewport(0.0, 0.0, 0.2, 0.2)

        self.orientation_widget.SetEnabled(True)

    def add_frame(self, transform=np.eye(4), length=1, line_width=5, frame_name=None):
        """Custom axes visualization for showing poses.
        Default vtkAxisActor cannot have a negative position, use this function instead.

        Args:
            transform:
            length: Axes length
            line_width: Line width
            frame_name: Frame name to display

        Returns:
            Tuple of axis actors
        """

        # Display poses with axes
        vtk_x_axis_actor = VtkLinesActor()
        vtk_x_axis_actor.set_lines([0, 0, 0], [length, 0, 0])
        vtk_x_axis_actor.GetProperty().SetColor((255, 0, 0))
        vtk_x_axis_actor.GetProperty().SetLineWidth(line_width)

        vtk_y_axis_actor = VtkLinesActor()
        vtk_y_axis_actor.set_lines([0, 0, 0], [0, length, 0])
        vtk_y_axis_actor.GetProperty().SetColor((0, 255, 0))
        vtk_y_axis_actor.GetProperty().SetLineWidth(line_width)

        vtk_z_axis_actor = VtkLinesActor()
        vtk_z_axis_actor.set_lines([0, 0, 0], [0, 0, length])
        vtk_z_axis_actor.GetProperty().SetColor((0, 0, 255))
        vtk_z_axis_actor.GetProperty().SetLineWidth(line_width)

        vtk_transform = vtk.vtkTransform()
        vtk_transform.SetMatrix(transform.flatten())
        vtk_x_axis_actor.SetUserTransform(vtk_transform)
        vtk_y_axis_actor.SetUserTransform(vtk_transform)
        vtk_z_axis_actor.SetUserTransform(vtk_transform)

        self.add_actor(vtk_x_axis_actor)
        self.add_actor(vtk_y_axis_actor)
        self.add_actor(vtk_z_axis_actor)

        if frame_name is not None:
            vtk_text_labels = VtkTextLabels()
            vtk_text_labels.set_text_labels([transform[0:3, 3]], [frame_name])
            self.renderer.AddActor(vtk_text_labels)

            return vtk_x_axis_actor, vtk_y_axis_actor, vtk_z_axis_actor, vtk_text_labels

        return vtk_x_axis_actor, vtk_y_axis_actor, vtk_z_axis_actor

    def add_lines(self, src_points, dst_points):
        """Add a lines actor to the renderer.

        Args:
            src_points: Start points (N, 3)
            dst_points: Destination points (N, 3)

        Returns:
            VtkLinesActor
        """
        vtk_lines_actor = VtkLinesActor()
        vtk_lines_actor.set_lines(src_points, dst_points)
        self.renderer.AddActor(vtk_lines_actor)
        return vtk_lines_actor

    def add_mesh(
        self,
        mesh: Mesh,
        pose: np.ndarray = np.eye(4, dtype=np.float32),
        opacity: float = 1.0,
        colour=None,
    ):
        """Add a VtkPlyActor to the renderer to display a Mesh object.

        Args:
            mesh: Mesh instance
            pose: Pose
            opacity: Mesh visualization opacity
            colour: Colour to draw the mesh

        Returns:
            VtkPlyActor
        """
        vtk_ply_actor = VtkPlyActor()
        vtk_ply_actor.set_poly_data(mesh.vtk_poly_data)
        vtk_ply_actor.set_transform(pose)
        vtk_ply_actor.GetProperty().SetOpacity(opacity)

        if colour is not None:
            vtk_ply_actor.GetProperty().SetColor(colour)

        self.renderer.AddActor(vtk_ply_actor)
        return vtk_ply_actor

    def add_pc(self, pc: np.ndarray, colours=None):
        """Add a point cloud (3, N) to the renderer

        Args:
            pc: Point cloud (3, N)
            colours: Point colours

        Returns:
            VtkPcActor
        """
        vtk_pc_actor = VtkPcActor(pc.T, colours)
        self.renderer.AddActor(vtk_pc_actor)
        return vtk_pc_actor

    def add_points(
        self,
        points: Union[List, np.ndarray],
        *,
        normals=None,
        colours: Optional[Union[list, np.ndarray]] = None,
        point_size=None,
        opacity=None,
    ) -> VtkPcActor:
        """Add a set of points (N, 3) to the renderer.
        If normals are provided, only the normals will be visualized.

        Args:
            points: Points (N, 3)
            normals: Normals
            colours: Point colours
            point_size: Point size
            opacity: Point opacity

        Returns:
            VtkPcActor
        """
        return vtk_utils.add_points(
            self.renderer,
            points,
            colours=colours,
            normals=normals,
            opacity=opacity,
            point_size=point_size,
        )

    def add_pose(self, transform=np.eye(4), length=1, line_width=5, frame_name=None):
        """Helper wrapper to call self.add_frame"""
        self.add_frame(transform, length, line_width, frame_name)

    def add_actor_toggles(self, actors):
        """Add display toggles for actors on F1-F12 keys"""
        self.interactor_style.actors = actors

    def init_cam(self, cam_p, focal_point=None):
        """Initialize the camera to match extrinsics"""
        vtk_utils.init_cam(self.camera, cam_p, focal_point)

    def show(self, reset_camera=False, blocking=True):
        """Show the render window"""
        self.renderer.ResetCameraClippingRange()
        self.render_window.Render()

        if reset_camera:
            self.renderer.ResetCamera()

        if blocking:
            self.interactor.Start()
        else:
            self.interactor.Initialize()
