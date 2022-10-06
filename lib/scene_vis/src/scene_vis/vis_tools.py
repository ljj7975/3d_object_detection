import cv2
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Window sizes
CV2_SIZE_2_COL = (930, 280)
CV2_SIZE_3_COL = (620, 187)
CV2_SIZE_4_COL = (465, 140)


def project_pc_to_image(point_cloud: np.ndarray, cam_p: np.ndarray):
    """Projects a 3D point cloud to 2D points

    Args:
        point_cloud: (3, N) point cloud
        cam_p: 3x4 camera projection matrix

    Returns:
        pts_2d: (2, N) projected coordinates [u, v] of the 3D points
    """

    pc_padded = np.append(point_cloud, np.ones((1, point_cloud.shape[1])), axis=0)
    pts_2d = np.dot(cam_p, pc_padded)

    pts_2d[0:2] = pts_2d[0:2] / pts_2d[2]
    return pts_2d[0:2]


def project_corners_3d_to_image(corners_3d, p):
    """Computes the 3D bounding box projected onto
    image space.

    Keyword arguments:
    obj -- object file to draw bounding box
    p -- transform matrix

    Returns:
        corners: (N, 8, 2) Corner points projected into image space
        face_idx: 3D bounding box faces for drawing
    """
    # index for 3d bounding box face
    # it is converted to 4x4 matrix
    face_idx = np.array(
        [0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7]  # front face  # left face  # back face
    ).reshape(
        (4, 4)
    )  # right face
    return project_pc_to_image(corners_3d, p), face_idx


def compute_box_3d_corners(box_3d):
    """Computes the 3D bounding box corner positions from a 3D box

    Args:
        box_3d: 3D box (x, y, z, l, w, h, ry)

    Returns:
        corners_3d:
    """

    tx, ty, tz, l, w, h, ry = box_3d

    # compute rotational matrix
    rot = np.array([[+np.cos(ry), 0, +np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, +np.cos(ry)]])

    # 3D BB corners
    x_corners = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    z_corners = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])

    corners_3d = np.dot(rot, np.array([x_corners, y_corners, z_corners]))

    corners_3d[0, :] = corners_3d[0, :] + tx
    corners_3d[1, :] = corners_3d[1, :] + ty
    corners_3d[2, :] = corners_3d[2, :] + tz

    return corners_3d


def project_orientation_3d(box_3d, cam_p):
    """Projects orientation vector given object and camera matrix

    Args:
        box_3d: 3D box (x, y, z, l, w, h, ry)
        cam_p: 3x4 camera projection matrix

    Returns:
        Projection of orientation vector into image
    """

    tx, ty, tz, l, w, h, ry = box_3d

    # Rotation matrix
    rot = np.array([[+np.cos(ry), 0, +np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, +np.cos(ry)]])

    orientation_3d = np.array([0.0, l, 0.0, 0.0, 0.0, 0.0]).reshape(3, 2)
    orientation_3d = np.dot(rot, orientation_3d)

    orientation_3d[0, :] = orientation_3d[0, :] + tx
    orientation_3d[1, :] = orientation_3d[1, :] + ty
    orientation_3d[2, :] = orientation_3d[2, :] + tz

    # only draw for boxes that are in front of the camera
    for idx in np.arange(orientation_3d.shape[1]):
        if orientation_3d[2, idx] < 0.1:
            return None

    return project_pc_to_image(orientation_3d, cam_p)


def plots_from_image(img, subplot_rows=1, subplot_cols=1, display=True, fig_size=None):
    """Forms the plot figure and axis for the visualization

    Args:
        img: image to plot
        subplot_rows: number of rows of the subplot grid
        subplot_cols: number of columns of the subplot grid
        display: display the image in non-blocking fashion
        fig_size: (optional) size of the figure
    """

    def set_plot_limits(axes, image):
        # Set the plot limits to the size of the image, y is inverted
        axes.set_xlim(0, image.shape[1])
        axes.set_ylim(image.shape[0], 0)

    if fig_size is None:
        img_shape = np.shape(img)
        fig_height = img_shape[0] / 100 * subplot_cols
        fig_width = img_shape[1] / 100 * subplot_rows
        fig_size = (fig_width, fig_height)

    # Create the figure
    fig, axes = plt.subplots(subplot_rows, subplot_cols, figsize=fig_size, sharex=True)
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, hspace=0.0)

    # Plot image
    if subplot_rows == 1 and subplot_cols == 1:
        # Single axis
        axes.imshow(img)
        set_plot_limits(axes, img)
    else:
        # Multiple axes
        for idx in range(axes.size):
            axes[idx].imshow(img)
            set_plot_limits(axes[idx], img)

    if display:
        plt.show(block=False)

    return fig, axes


def plots_from_sample_name(
    image_dir, sample_name, subplot_rows=1, subplot_cols=1, display=True, fig_size=(15, 9.15)
):
    """Forms the plot figure and axis for the visualization

    Args:
        image_dir: directory of image files in the wavedata
        sample_name: sample name of the image file to present
        subplot_rows: number of rows of the subplot grid
        subplot_cols: number of columns of the subplot grid
        display: display the image in non-blocking fashion
        fig_size: (optional) size of the figure
    """
    sample_name = int(sample_name)

    # Grab image data
    img = np.array(Image.open("{}/{:06d}.png".format(image_dir, sample_name)), dtype=np.uint8)

    # Create plot
    fig, axes = plots_from_image(img, subplot_rows, subplot_cols, display, fig_size)

    return fig, axes


class ZoomPan:
    """Add zoom and pan functionality to matplotlib plots
    Modified from https://stackoverflow.com/a/19829987/13688973
    Controls:
        Scroll wheel - Zoom in/out
        Mouse drag - Pan
        Ctrl + q - Close all windows
    """

    def __init__(self):
        self.press = None
        self.cur_xlim = None
        self.cur_ylim = None
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.xpress = None
        self.ypress = None

        self.init_xlim = None
        self.init_ylim = None

    def zoom_factory(self, ax, base_scale=1.2):
        def zoom(event):
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()

            xdata = event.x  # get event x location
            ydata = event.y  # get event y location

            if event.button == "up":
                # deal with zoom out
                scale_factor = 1 / base_scale
            elif event.button == "down":
                # deal with zoom in
                scale_factor = base_scale
            else:
                # deal with something that should never happen
                scale_factor = 1
                print(event.button)

            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

            relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

            new_x1 = xdata - new_width * (1 - relx)
            new_x2 = xdata + new_width * relx
            new_y1 = ydata - new_height * (1 - rely)
            new_y2 = ydata + new_height * rely
            if new_x1 < 0 and new_x2 < self.init_xlim[1]:
                # Shift left
                shift_amount = (self.init_xlim[1] - new_x2) / 3.0
                new_x2 += shift_amount
                new_x1 += shift_amount
            if new_x2 > self.init_xlim[1] and new_x1 > 0:
                # Shift right
                shift_amount = new_x1 / 3.0
                new_x1 -= shift_amount
                new_x2 -= shift_amount

            ax.set_xlim([new_x1, new_x2])
            ax.set_ylim([new_y1, new_y2])
            ax.figure.canvas.draw()

        fig = ax.get_figure()
        fig.canvas.mpl_connect("scroll_event", zoom)
        self.init_xlim = ax.get_xlim()
        self.init_ylim = ax.get_ylim()

        return zoom

    def pan_factory(self, ax):
        def on_press(event):
            if event.inaxes != ax:
                return
            self.cur_xlim = ax.get_xlim()
            self.cur_ylim = ax.get_ylim()
            self.press = self.x0, self.y0, event.xdata, event.ydata
            self.x0, self.y0, self.xpress, self.ypress = self.press

        def on_release(event):
            self.press = None
            ax.figure.canvas.draw()

        def on_motion(event):
            if self.press is None:
                return
            if event.inaxes != ax:
                return
            dx = event.xdata - self.xpress
            dy = event.ydata - self.ypress
            self.cur_xlim -= dx
            self.cur_ylim -= dy
            ax.set_xlim(self.cur_xlim)
            ax.set_ylim(self.cur_ylim)

            ax.figure.canvas.draw()

        fig = ax.get_figure()  # get the figure of interest

        # attach the call back
        fig.canvas.mpl_connect("button_press_event", on_press)
        fig.canvas.mpl_connect("button_release_event", on_release)
        fig.canvas.mpl_connect("motion_notify_event", on_motion)

        # return the function
        return on_motion

    def key_factory(self, ax):
        def on_keypress(event):
            if event.key == "ctrl+q":
                print("close all")
                plt.close("all")

        fig = ax.get_figure()
        fig.canvas.mpl_connect("key_press_event", on_keypress)


def move_plt_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == "TkAgg":
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == "WXAgg":
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)


def _set_plot_limits(axes, image):
    # Set the plot limits to the size of the image
    axes.set_xlim(0, image.shape[1])
    axes.set_ylim(0, image.shape[0])


def _calc_window_location(row_col, size_wh, start_x, start_y, y_offset):
    """Calculates the plot window location"""

    subplot_row = row_col[0]
    subplot_col = row_col[1]
    location_xy = (
        int(start_x + subplot_col * size_wh[0]),
        int(start_y + subplot_row * size_wh[1] + subplot_row * y_offset),
    )
    return location_xy


def cv2_imshow(window_name, image, size_wh=None, row_col=None, location_xy=None):
    """Helper function for specifying window size and location when
        displaying images with cv2

    Args:
        window_name (string): Window title
        image: image to display
        size_wh: resize window
            Recommended sizes for 1920x1080 screen:
                2 col: (930, 280)
                3 col: (620, 187)
                4 col: (465, 140)
        row_col: Row and column to show images like subplots
        location_xy: location of window
    """
    if size_wh is None:
        size_wh = image.shape[0:2][::-1]

    try:
        if size_wh is not None:
            cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow(window_name, *size_wh)
        else:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        # Calculate desired window location
        if row_col is not None:
            location_xy = _calc_window_location(
                row_col, size_wh, start_x=60, start_y=25, y_offset=28
            )

        if location_xy is not None:
            cv2.moveWindow(window_name, *location_xy)

        cv2.imshow(window_name, image)

    except Exception:
        # Fall back to matplotlib if cv.imshow is not available
        import matplotlib.pyplot as plt

        if len(image.shape) == 2:
            # Convert single channel image to BGR, otherwise will be flipped horizontally
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        matplotlib.rcParams["toolbar"] = "None"
        fig, axes = plt.subplots(1, 1, figsize=np.asarray(size_wh) / 100.0, sharex=True)
        fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, hspace=0.0)
        fig.canvas.set_window_title(window_name)
        fig.canvas.toolbar_visible = False

        axes.axis("off")
        axes.imshow(image[..., ::-1], origin="upper")  # expecting BGR image
        # set_plot_limits(axes, image)

        # Calculate desired window location
        if row_col is not None:
            location_xy = _calc_window_location(
                row_col, size_wh, start_x=55, start_y=0, y_offset=40
            )

        if location_xy is not None:
            move_plt_figure(fig, *location_xy)

        zp = ZoomPan()
        zp.zoom_factory(axes)
        zp.pan_factory(axes)
        zp.key_factory(axes)

        plt.show(block=False)

        return fig


def get_point_colours(points, cam_p, image):
    points_in_im = project_pc_to_image(points.T, cam_p)
    points_in_im_rounded = np.round(points_in_im).astype(np.int32)

    point_colours = image[points_in_im_rounded[1], points_in_im_rounded[0]]

    return point_colours


def draw_box_2d(ax, box_2d, color="#90EE900", linewidth=2):
    """Draws 2D boxes given coordinates in box_2d format

    Args:
        ax: subplot handle
        box_2d: ndarray containing box coordinates in box_2d format (y1, x1, y2, x2)
        color: color of box
    """
    box_x1 = box_2d[1]
    box_y1 = box_2d[0]
    box_w = box_2d[3] - box_x1
    box_h = box_2d[2] - box_y1

    rect = patches.Rectangle(
        (box_x1, box_y1), box_w, box_h, linewidth=linewidth, edgecolor=color, facecolor="none"
    )
    ax.add_patch(rect)
