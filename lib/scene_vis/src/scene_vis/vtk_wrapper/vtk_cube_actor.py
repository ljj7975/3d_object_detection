
import vtk

from nova.utils import matrix_utils


class VtkCubeActor(vtk.vtkActor):
    """Cube actor to represent boxes, only supports rotation around z-axis
    """

    def __init__(self, position, lengths, rz, color=None, opacity:float=None):
        self.set_cube(position, lengths, rz)
        if color is not None:
            self.set_color(color)
        if opacity is not None:
            self.set_opacity(opacity)

    def set_cube(self, position, lengths, rz):
        vtk_cube_source = vtk.vtkCubeSource()
        vtk_cube_source.SetXLength(lengths[0])
        vtk_cube_source.SetYLength(lengths[1])
        vtk_cube_source.SetZLength(lengths[2])
        vtk_cube_mapper = vtk.vtkPolyDataMapper()
        vtk_cube_mapper.SetInputConnection(vtk_cube_source.GetOutputPort())

        tfm_mat = matrix_utils.get_rot_z_tfm(rz)
        tfm_mat[0:3, 3] = position

        vtk_cube_transform = vtk.vtkTransform()
        vtk_cube_transform.SetMatrix(tfm_mat.flatten())

        self.SetMapper(vtk_cube_mapper)
        self.GetProperty().SetOpacity(0.1)
        self.GetProperty().SetLineWidth(5)
        self.SetUserTransform(vtk_cube_transform)

    def set_transform(self, transform):
        self.vtk_transform = vtk.vtkTransform()
        self.vtk_transform.SetMatrix(transform.flatten())
        self.SetUserTransform(self.vtk_transform)

    def set_color(self, color):
        self.GetProperty().SetColor(color)

    def set_opacity(self, opacity):
        self.GetProperty().SetOpacity(opacity)
