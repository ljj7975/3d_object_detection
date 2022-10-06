"""https://github.com/kujason/scene_vis"""

import datetime
import os

import numpy as np
import vtk
from scene_vis.vtk_wrapper.vtk_lines_actor import VtkLinesActor
from scene_vis.vtk_wrapper.vtk_pc_actor import VtkPcActor
from scene_vis.vtk_wrapper.vtk_text_labels import VtkTextLabels
from vtk.util import numpy_support

COLOUR_SCHEME_KITTI = {
    "Car": (255, 0, 0),  # Red
    "Pedestrian": (255, 150, 50),  # Orange
    "Cyclist": (150, 50, 100),  # Purple
    "Van": (255, 150, 150),  # Peach
    "Person_sitting": (150, 200, 255),  # Sky Blue
    "Truck": (200, 200, 200),  # Light Grey
    "Tram": (150, 150, 150),  # Grey
    "Misc": (100, 100, 100),  # Dark Grey
    "DontCare": (255, 255, 255),  # White
}


class ToggleActorsInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    """VTK interactor style that allows toggling the visibility of up to 12
    actors with the F1-F12 keys. This object should be initialized with
    the actors to toggle, and each actor will be assigned an F key based on
    its order in the list.
    """

    def __init__(
        self,
        actors,
        vtk_renderer,
        current_cam=None,
        axes=None,
        focal_point=(0.0, 0.0, 1000.0),
        vtk_win_to_img_filter=None,
        vtk_png_writer=None,
    ):
        super().__init__()

        self.actors = actors
        self.AddObserver("KeyPressEvent", self.key_press_event)
        self.vtk_renderer = vtk_renderer
        self.current_cam = current_cam
        self.axes = axes
        self.focal_point = focal_point

        # Screenshots
        self.vtk_win_to_img_filter = vtk_win_to_img_filter
        self.vtk_png_writer = vtk_png_writer

    def key_press_event(self, obj, event):
        vtk_render_window_interactor = self.GetInteractor()

        key = vtk_render_window_interactor.GetKeySym()
        if key in ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12"]:

            actor_idx = int(key.split("F")[1]) - 1

            if self.actors and actor_idx < len(self.actors) and self.actors[actor_idx] is not None:
                current_prop = self.actors[actor_idx]
                if isinstance(current_prop, vtk.vtkCollection):
                    for actor in current_prop:
                        current_visibility = actor.GetVisibility()
                        actor.SetVisibility(not current_visibility)
                else:
                    current_visibility = current_prop.GetVisibility()
                    current_prop.SetVisibility(not current_visibility)
                self.vtk_renderer.GetRenderWindow().Render()

        elif key == "t":
            if self.vtk_renderer is not None and self.vtk_renderer.GetActiveCamera() is not None:
                camera = self.vtk_renderer.GetActiveCamera()
                camera.SetViewUp(0, -1, 0)
                camera.SetPosition(0, 0, 0)
                camera.SetFocalPoint(*self.focal_point)
                self.vtk_renderer.ResetCameraClippingRange()
                self.vtk_renderer.GetRenderWindow().Render()

        elif key == "a":
            if self.axes is not None:
                current_visibility = self.axes.GetVisibility()
                self.axes.SetVisibility(not current_visibility)
                self.vtk_renderer.GetRenderWindow().Render()

        elif key == "c":
            camera = self.vtk_renderer.GetActiveCamera()
            print("Cam Position: {:.3f}, {:.3f}, {:.3f}".format(*camera.GetPosition()))
            print("Cam Focal Point: {:.3f}, {:.3f}, {:.3f}".format(*camera.GetFocalPoint()))
            print("Cam View Up: {:.3f}, {:.3f}, {:.3f}".format(*camera.GetViewUp()))

            curr_view_tfm = np_from_vtk4x4(camera.GetViewTransformMatrix())
            print(curr_view_tfm)

        elif key == "p":
            if self.vtk_win_to_img_filter is not None:
                now = datetime.datetime.now()
                screenshot_name = "screenshot_{}.png".format(now)
                save_screenshot(screenshot_name, self.vtk_win_to_img_filter, self.vtk_png_writer)
                print("Saved", os.getcwd() + "/" + screenshot_name)


def setup_renderer_and_window(
    background=(0.2, 0.3, 0.4),
    window_title="VTK",
    window_size=(960, 600),
    offscreen_rendering=False,
):

    vtk_renderer = vtk.vtkRenderer()
    vtk_renderer.SetBackground(*background)

    # Setup Render Window
    vtk_render_window = vtk.vtkRenderWindow()
    vtk_render_window.SetWindowName(window_title)
    vtk_render_window.SetSize(*window_size)

    if offscreen_rendering:
        vtk_render_window.SetOffScreenRendering(offscreen_rendering)

    vtk_render_window.AddRenderer(vtk_renderer)

    return vtk_renderer, vtk_render_window


def setup_interactor_style(vtk_actors, vtk_renderer):
    interactor_style = ToggleActorsInteractorStyle(vtk_actors, vtk_renderer)
    return interactor_style


def setup_interactor(vtk_render_window):
    vtk_interactor = vtk.vtkRenderWindowInteractor()
    vtk_interactor.SetRenderWindow(vtk_render_window)

    return vtk_interactor


def set_interactor_style(vtk_interactor, interactor_style):
    vtk_interactor.SetInteractorStyle(interactor_style)


def set_axes_font_size(axes, font_size):

    axes.GetXAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
    axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(font_size)
    axes.GetYAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
    axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(font_size)
    axes.GetZAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
    axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(font_size)


def add_axes_widget(vtk_interactor, viewport_bounds=(0.0, 0.0, 0.2, 0.2)):
    """Adds an axes actor as a widget in the specified viewport area.

    Notes:
        The returned axes and orientation_widget must be stored in a reference,
        otherwise there will be a SIGSEV

    Args:
        vtk_interactor: Interactor to match axes orientation during rotation
        viewport_bounds: Viewport bounds (Default bottom left)

    Returns:
        axes: vtkAxesActor
        orientation_widget: vtkOrientationMarkerWidget
    """
    axes = vtk.vtkAxesActor()
    orientation_widget = vtk.vtkOrientationMarkerWidget()
    orientation_widget.SetOrientationMarker(axes)
    orientation_widget.SetInteractor(vtk_interactor)
    orientation_widget.SetViewport(*viewport_bounds)
    orientation_widget.SetEnabled(True)
    return axes, orientation_widget


def setup_screenshots(vtk_render_window, img_type="png"):
    """Sets up window to image filter, and png writer

    Args:
        vtk_render_window: Instance of vtkRenderWindow

    Returns:
        vtk_win_to_img_filter: Instance of vtkWindowToImageFilter
        vtk_img_writer: Instance of vtkPNGWriter or vtkJPEGWriter
    """
    vtk_win_to_img_filter = vtk.vtkWindowToImageFilter()
    vtk_win_to_img_filter.SetInput(vtk_render_window)

    if img_type == "png":
        vtk_img_writer = vtk.vtkPNGWriter()
    elif img_type == "jpg":
        vtk_img_writer = vtk.vtkJPEGWriter()
    else:
        raise ValueError(f"Invalid img_type: {img_type}, should be png or jpg")

    return vtk_win_to_img_filter, vtk_img_writer


def take_screenshot(vtk_win_to_img_filter):
    """
    https://stackoverflow.com/questions/14553523/vtk-render-window-image-to-numpy-array/14568878#14568878
    """

    vtk_win_to_img_filter.Modified()
    vtk_win_to_img_filter.Update()

    vtk_image = vtk_win_to_img_filter.GetOutput()
    width, height, _ = vtk_image.GetDimensions()
    vtk_array = vtk_image.GetPointData().GetScalars()
    n_components = vtk_array.GetNumberOfComponents()
    image = numpy_support.vtk_to_numpy(vtk_array).reshape(height, width, n_components)

    # Image is flipped vertically and RGB
    return image[::-1][..., ::-1]


def save_screenshot(file_path, vtk_win_to_img_filter, vtk_img_writer):
    """Saves a screenshot of the current render window

    Args:
        file_path: File path
        vtk_win_to_img_filter: Instance of vtkWindowToImageFilter
        vtk_img_writer: Instance of vtkPNGWriter or vtkJPEGWriter
    """
    # Update filter
    vtk_win_to_img_filter.Modified()
    vtk_win_to_img_filter.Update()

    # Check file type
    file_ext = os.path.splitext(file_path)[-1]
    if isinstance(vtk_img_writer, vtk.vtkPNGWriter) and file_ext != ".png":
        raise ValueError("File extension should be .png for vtk.vtkPNGWriter")
    if isinstance(vtk_img_writer, vtk.vtkJPEGWriter) and file_ext != ".jpg":
        raise ValueError("File extension should be .jpg for vtk.vtkJPEGWriter")

    # Save image
    vtk_img_writer.SetFileName(file_path)
    vtk_img_writer.SetInputData(vtk_win_to_img_filter.GetOutput())
    vtk_img_writer.Write()


def simulate_camera(vtk_camera, image_shape, cam_mat):

    # Set window center to offset principal point
    h, w = image_shape[0:2]
    cx = cam_mat[0, 2]
    cy = cam_mat[1, 2]
    fx = fy = cam_mat[0, 0]
    # fy = cam_mat[1, 1]
    wcx = -2.0 * (cx - w / 2.0) / w
    wcy = 2.0 * (cy - h / 2.0) / h
    vtk_camera.SetWindowCenter(wcx, wcy)

    # Set vertical view angle as a indirect way of setting the y focal distance
    view_angle = np.rad2deg(2.0 * np.arctan2(h / 2.0, fy))
    vtk_camera.SetViewAngle(view_angle)

    # Set the image aspect ratio as an indirect way of setting the x focal distance
    m = np.eye(4)
    aspect = fy / fx
    m[0, 0] = 1.0 / aspect
    t = vtk.vtkTransform()
    t.SetMatrix(m.flatten())
    vtk_camera.SetUserTransform(t)

    vtk_camera.SetViewUp(0, -1, 0)
    vtk_camera.SetPosition(0, 0, 0)
    vtk_camera.SetFocalPoint(0, 0, fx)


def init_cam(vtk_camera, cam_p, focal_point=None):
    cam_f = cam_p[0, 0]
    vtk_camera.SetPosition(0, 0, 0)
    vtk_camera.SetViewUp(0, -1, 0)

    if focal_point is None:
        vtk_camera.SetFocalPoint(0, 0, cam_f)
    else:
        vtk_camera.SetFocalPoint(*focal_point)


def np_from_vtk4x4(vtk4x4):
    vals = [0.0] * 16
    vtk4x4.DeepCopy(vals, vtk4x4)
    return np.reshape(vals, [4, 4])


def add_points(
    vtk_renderer, points, *, colours=None, normals=None, opacity=None, point_size=None
) -> VtkPcActor:

    points = np.asarray(points)
    if len(points.shape) != 2 or points.shape[1] != 3:
        raise ValueError(f"Invalid shape for points, should be (N, 3), is {points.shape}")

    vtk_pc_actor = VtkPcActor(points, colours)
    if normals is not None:
        vtk_pc_actor.set_normals(normals)
    if opacity is not None:
        vtk_pc_actor.GetProperty().SetOpacity(opacity)
    if point_size is not None:
        vtk_pc_actor.set_point_size(point_size)
    vtk_renderer.AddActor(vtk_pc_actor)
    return vtk_pc_actor


def add_vtk_sphere(vtk_renderer, point, sphere_colour, radius):
    vtk_sphere_source = vtk.vtkSphereSource()
    vtk_sphere_source.SetCenter(*point)
    vtk_sphere_source.SetRadius(radius)
    vtk_sphere_mapper = vtk.vtkPolyDataMapper()
    vtk_sphere_mapper.SetInputConnection(vtk_sphere_source.GetOutputPort())
    vtk_sphere_actor = vtk.vtkActor()
    vtk_sphere_actor.SetMapper(vtk_sphere_mapper)

    vtk_sphere_actor.GetProperty().SetColor(sphere_colour)
    vtk_renderer.AddActor(vtk_sphere_actor)

    return vtk_sphere_actor


def add_frame(vtk_renderer, transform=np.eye(4), length=1, line_width=5, frame_name=None):
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

    vtk_renderer.AddActor(vtk_x_axis_actor)
    vtk_renderer.AddActor(vtk_y_axis_actor)
    vtk_renderer.AddActor(vtk_z_axis_actor)

    if frame_name is not None:
        vtk_text_labels = VtkTextLabels()
        vtk_text_labels.set_text_labels([transform[0:3, 3]], [frame_name])
        vtk_renderer.AddActor(vtk_text_labels)

        return vtk_x_axis_actor, vtk_y_axis_actor, vtk_z_axis_actor, vtk_text_labels

    return vtk_x_axis_actor, vtk_y_axis_actor, vtk_z_axis_actor


def collection_from_actor_list(actor_list):
    collection = vtk.vtkActorCollection()
    for actor in actor_list:
        collection.AddItem(actor)
    return collection
