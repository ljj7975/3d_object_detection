"""https://github.com/kujason/scene_vis"""

import numpy as np
import vtk
from vtk.util import numpy_support


class VtkPcActor(vtk.vtkActor):
    def __init__(self, points=None, point_colours=None, point_size=None, opacity=None):
        super().__init__()

        # References to the converted numpy arrays to avoid seg faults
        self.np_to_vtk_points = None
        self.np_to_vtk_cells = None

        # Point data
        self.points = None

        # VTK Data
        self.vtk_poly_data = vtk.vtkPolyData()
        self.vtk_points = vtk.vtkPoints()
        self.vtk_cells = vtk.vtkCellArray()

        # Colours for each point in the point cloud
        self.vtk_colours = None
        # self.vtk_colours = vtk.vtkUnsignedCharArray()
        # self.vtk_colours.SetNumberOfComponents(3)
        # self.vtk_colours.SetName("Colours")

        # Poly Data
        self.vtk_poly_data.SetPoints(self.vtk_points)
        self.vtk_poly_data.SetVerts(self.vtk_cells)

        # Poly Data Mapper
        self.vtk_poly_data_mapper = vtk.vtkPolyDataMapper()
        self.vtk_poly_data_mapper.SetInputData(self.vtk_poly_data)

        self.vtk_transform = None

        # Actor
        self.SetMapper(self.vtk_poly_data_mapper)

        if points is not None:
            if point_colours is not None:
                self.set_points(points, point_colours)
            else:
                self.set_points(points)
        if point_size is not None:
            self.set_point_size(point_size)
        if opacity is not None:
            self.set_opacity(opacity)

    def set_pc(self, pc, point_colours=None):
        """Set the point cloud to be visualized

        Args:
            pc: (3, N) Point cloud
            point_colours: (N, 3) List of BGR colours as uint8
        """
        self.set_points(pc.T, point_colours)

    def set_points(self, points, point_colours=None):
        """Sets the points to be visualized

        Args:
            points: (N, 3) List of points
            point_colours: (N x 3) List of BGR colours as uint8
        """

        num_points = len(points)
        if num_points == 0:
            return

        # Set the points
        self.points = points
        flattened_points = np.asarray(points).flatten().astype(np.float32)
        self.np_to_vtk_points = numpy_support.numpy_to_vtk(
            flattened_points, deep=True, array_type=vtk.VTK_TYPE_FLOAT32
        )
        self.np_to_vtk_points.SetNumberOfComponents(3)
        self.vtk_points.SetData(self.np_to_vtk_points)

        # Create cells, one per point, cells in the form: [length, point index]
        cell_lengths = np.ones(num_points)
        cell_indices = np.arange(0, num_points)
        flattened_cells = np.array([cell_lengths, cell_indices]).transpose().flatten()
        flattened_cells = flattened_cells.astype(np.int32)

        # Convert list of cells to vtk format and set the cells
        self.np_to_vtk_cells = numpy_support.numpy_to_vtk(
            flattened_cells, deep=True, array_type=vtk.VTK_ID_TYPE
        )
        self.np_to_vtk_cells.SetNumberOfComponents(2)
        self.vtk_cells.SetCells(num_points, self.np_to_vtk_cells)

        self.set_point_colours(point_colours)

    def set_normals(
        self, normals, tip_resolution=8, tip_length=0.2, tip_radius=0.15, scale_factor=3.0
    ):

        vtk_float_array = numpy_support.numpy_to_vtk(normals, True)
        self.vtk_poly_data.GetPointData().SetNormals(vtk_float_array)

        arrow_source = vtk.vtkArrowSource()
        arrow_source.SetTipResolution(tip_resolution)
        arrow_source.SetTipLength(tip_length)
        arrow_source.SetTipRadius(tip_radius)

        glyph_3d = vtk.vtkGlyph3D()
        glyph_3d.SetSourceConnection(arrow_source.GetOutputPort())
        glyph_3d.SetInputData(self.vtk_poly_data)
        glyph_3d.SetVectorModeToUseNormal()
        glyph_3d.SetScaleModeToScaleByVector()
        glyph_3d.SetScaleFactor(scale_factor)
        glyph_3d.OrientOn()
        glyph_3d.Update()

        self.vtk_poly_data_mapper.SetInputConnection(glyph_3d.GetOutputPort())

    def set_point_colours(self, point_colours: np.ndarray, axis="z"):
        """Set point colours

        Args:
            point_colours:

        Returns:

        """
        if point_colours is not None:
            if isinstance(point_colours, list) and len(point_colours) == 3:
                point_colours = np.full_like(self.points, point_colours, dtype=np.uint8)

            if point_colours.dtype != np.uint8:
                point_colours = (point_colours * 255).astype(np.uint8)

            if len(point_colours) != len(self.points):
                raise ValueError(
                    f"Length of point_colours ({len(point_colours)}) should match number of points ({len(self.points)})"
                )

            # Set point colours if provided
            # Rearrange OpenCV BGR into RGB format
            point_colours = np.asarray(point_colours)[:, [2, 1, 0]]

            # Set the point colours
            flattened_colours = np.asarray(point_colours).flatten()
            self.vtk_colours = numpy_support.numpy_to_vtk(
                flattened_colours, deep=True, array_type=vtk.VTK_TYPE_UINT8
            )
            self.vtk_colours.SetNumberOfComponents(3)

        else:
            # Use coordinates along default_axis if no colours provided
            if axis == "z":
                axis_idx = 2
            elif axis == "y":
                axis_idx = 1
            elif axis == "x":
                axis_idx = 0
            else:
                raise ValueError("Invalid axis", axis)

            val_min = np.amin(self.points, axis=0)[axis_idx]
            val_max = np.amax(self.points, axis=0)[axis_idx]
            val_range = val_max - val_min

            if val_range > 0:
                pts = (self.points.transpose()[axis_idx] - val_min) / val_range
            else:
                pts = val_min

            scalar_array = pts.astype(np.float32)
            self.vtk_colours = numpy_support.numpy_to_vtk(
                scalar_array, deep=True, array_type=vtk.VTK_TYPE_FLOAT32
            )
            self.vtk_colours.SetNumberOfComponents(1)

            # Update PolyDataMapper to display height scalars
            self.vtk_poly_data_mapper.SetColorModeToDefault()
            self.vtk_poly_data_mapper.SetScalarRange(0, 1.0)
            self.vtk_poly_data_mapper.SetScalarVisibility(1)

        # Set point colours in Poly Data
        self.vtk_poly_data.GetPointData().SetScalars(self.vtk_colours)

    def set_point_size(self, point_size):
        self.GetProperty().SetPointSize(point_size)

    def set_opacity(self, opacity):
        self.GetProperty().SetOpacity(opacity)

    def set_transform(self, transform):
        self.vtk_transform = vtk.vtkTransform()
        self.vtk_transform.SetMatrix(transform.flatten())
        self.SetUserTransform(self.vtk_transform)

    def add_to_renderer(self, vtk_renderer):
        vtk_renderer.AddActor(self)

    def remove_from_renderer(self, vtk_renderer):
        vtk_renderer.RemoveActor(self)
