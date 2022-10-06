"""https://github.com/kujason/scene_vis"""

import numpy as np
import vtk
from vtk.util import numpy_support


class VtkLinesActor(vtk.vtkActor):
    def __init__(self):
        super().__init__()

        # References to the converted numpy arrays to avoid seg faults
        self.np_to_vtk_points = None
        self.np_to_vtk_cells = None

        # Point data
        self.src_points = None
        self.dst_points = None

        # VTK Data
        self.vtk_poly_data = vtk.vtkPolyData()

        self.vtk_points = vtk.vtkPoints()
        # self.vtk_dst_points = vtk.vtkPoints()

        self.vtk_cells = vtk.vtkCellArray()

        # Colours for each point in the point cloud
        self.vtk_colours = None
        # self.vtk_colours = vtk.vtkUnsignedCharArray()
        # self.vtk_colours.SetNumberOfComponents(3)
        # self.vtk_colours.SetName("Colours")

        # Poly Data
        self.vtk_poly_data.SetLines(self.vtk_cells)
        self.vtk_poly_data.SetPoints(self.vtk_points)
        # self.vtk_poly_data.SetVerts(self.vtk_cells)

        # Poly Data Mapper
        self.vtk_poly_data_mapper = vtk.vtkPolyDataMapper()
        self.vtk_poly_data_mapper.SetInputData(self.vtk_poly_data)

        # Actor
        self.SetMapper(self.vtk_poly_data_mapper)

    def set_pc(self, pc, point_colours=None):
        """Set the point cloud to be visualized

        Args:
            pc: (3, N) Point cloud 
            point_colours: (N, 3) List of BGR colours as uint8
        """
        self.set_points(pc.T, point_colours)

    def set_lines(self, src_points, dst_points, line_colours=None):
        """Sets the points to be visualized

        Args:
            src_points: (N, 3) List of start points
            dst_points: (N, 3) List of end points
            line_colours: (N x 3) List of BGR colours as uint8 
        """

        num_lines = len(src_points)

        # num_points = len(src_points) + len(dst_points)

        # Set the points
        self.src_points = src_points
        self.dst_points = dst_points

        flattened_points = np.concatenate(
            [src_points, dst_points], axis=0).flatten().astype(np.float32)
        self.np_to_vtk_points = numpy_support.numpy_to_vtk(
            flattened_points, deep=True, array_type=vtk.VTK_TYPE_FLOAT32)
        self.np_to_vtk_points.SetNumberOfComponents(3)
        self.vtk_points.SetData(self.np_to_vtk_points)

        # Create cells, one per point, cells in the form: [length, point index]
        for i in range(len(src_points)):
            vtk_line = vtk.vtkLine()
            vtk_line.GetPointIds().SetId(0, i)
            vtk_line.GetPointIds().SetId(1, num_lines + i)

            self.vtk_cells.InsertNextCell(vtk_line)

        # TODO: Use numpy_support
        # cell_lengths = np.ones(num_points)
        # cell_indices = np.arange(0, num_points)
        # flattened_cells = np.array(
        #     [cell_lengths, cell_indices]).transpose().flatten()
        # flattened_cells = flattened_cells.astype(np.int32)
        #
        # # Convert list of cells to vtk format and set the cells
        # self.np_to_vtk_cells = numpy_support.numpy_to_vtk(
        #     flattened_cells, deep=True, array_type=vtk.VTK_ID_TYPE)
        # self.np_to_vtk_cells.SetNumberOfComponents(2)
        # self.vtk_cells.SetCells(num_points, self.np_to_vtk_cells)

        # self.set_point_colours(line_colours)

    def set_line_width(self, line_width):
        self.GetProperty().SetLineWidth(line_width)

    def set_point_colours(self, point_colours, axis='z'):
        """Set point colours

        Args:
            point_colours: 

        Returns:

        """
        if point_colours is not None:
            # Set point colours if provided
            # Rearrange OpenCV BGR into RGB format
            point_colours = np.asarray(point_colours)[:, [2, 1, 0]]

            # Set the point colours
            flattened_colours = np.asarray(point_colours).flatten()
            self.vtk_colours = numpy_support.numpy_to_vtk(
                flattened_colours, deep=True, array_type=vtk.VTK_TYPE_UINT8)
            self.vtk_colours.SetNumberOfComponents(3)

        else:
            # Use coordinates along default_axis if no colours provided
            if axis == 'z':
                axis_idx = 2
            elif axis == 'y':
                axis_idx = 1
            elif axis == 'x':
                axis_idx = 0
            else:
                raise ValueError('Invalid axis', axis)

            val_min = np.amin(self.src_points, axis=0)[axis_idx]
            val_max = np.amax(self.src_points, axis=0)[axis_idx]
            val_range = val_max - val_min

            if val_range > 0:
                pts = (self.src_points.transpose()[axis_idx] - val_min) / val_range
            else:
                pts = val_min

            scalar_array = pts.astype(np.float32)
            self.vtk_colours = numpy_support.numpy_to_vtk(
                scalar_array, deep=True, array_type=vtk.VTK_TYPE_FLOAT32)
            self.vtk_colours.SetNumberOfComponents(1)

            # Update PolyDataMapper to display height scalars
            self.vtk_poly_data_mapper.SetColorModeToDefault()
            self.vtk_poly_data_mapper.SetScalarRange(0, 1.0)
            self.vtk_poly_data_mapper.SetScalarVisibility(1)

        # Set point colours in Poly Data
        self.vtk_poly_data.GetPointData().SetScalars(self.vtk_colours)

    def set_point_size(self, point_size):
        self.GetProperty().SetPointSize(point_size)

    def add_to_renderer(self, vtk_renderer):
        vtk_renderer.AddActor(self)