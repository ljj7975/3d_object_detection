import numpy as np
import vtk


class VtkStackedBox(vtk.vtkActor):

    BOX_COLOURS = {
        "box1": np.asarray([255, 0, 255]),  # magenta
        "box2": np.asarray([255, 128, 0]),  # orange
        "box3": np.asarray([0, 0, 255]),  # blue
        "box4": np.asarray([0, 255, 0]),  # green

        "box1v": np.asarray([255, 0, 255]),  # magenta
        "box3v": np.asarray([0, 0, 255]),  # blue
        "box4v": np.asarray([0, 255, 0]),  # green
    }

    def __init__(self):
        super().__init__()

    def set_stacked_box(self, box, colour=None, opacity=None):

        box_id, row, col, height, rotated = box.location

        dx, dy = box.dx, box.dy
        if rotated:
            dy, dx = box.dx, box.dy

        # Padding between boxes
        sep = 0.2
        x1 = col + (sep * col)
        x2 = x1 + dx + (sep * (dx - 1))
        y1 = row + (sep * row)
        y2 = y1 + dy + (sep * (dy - 1))
        z1, z2 = -(height + box.dz), -height

        vtk_cube_source = vtk.vtkCubeSource()
        vtk_cube_source.SetBounds(x1, x2, y1, y2, z1, z2)

        vtk_poly_data_mapper = vtk.vtkPolyDataMapper()
        vtk_poly_data_mapper.SetInputConnection(vtk_cube_source.GetOutputPort())

        if colour is not None:
            self.GetProperty().SetColor(colour)
        else:
            self.GetProperty().SetColor(VtkStackedBox.BOX_COLOURS[box.box_type] / 255.0)

        if opacity is not None:
            self.GetProperty().SetOpacity(opacity)

        self.GetProperty().SetLineWidth(5.0)
        self.GetProperty().SetEdgeColor(0.0, 0.0, 0.0)  # Black
        self.GetProperty().EdgeVisibilityOn()

        self.SetMapper(vtk_poly_data_mapper)

    def set_transform(self, transform):
        vtk_transform = vtk.vtkTransform()
        vtk_transform.SetMatrix(transform.flatten())
        self.SetUserTransform(vtk_transform)

    def set_offsets(self, offsets):
        if offsets is not None:
            transform = np.eye(4)
            transform[0:3, 3] = offsets
            self.set_transform(transform)
