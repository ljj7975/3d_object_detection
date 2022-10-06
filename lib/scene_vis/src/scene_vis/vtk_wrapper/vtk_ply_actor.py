"""https://github.com/kujason/scene_vis"""
import os

import vtk


class VtkPlyActor(vtk.vtkActor):
    """Ply mesh actor. If opacity is not 1, vtk_renderer should call SetUseDepthPeeling(True)"""

    def __init__(self):
        super().__init__()

        # Use VTK PLY reader to keep mesh faces
        self.ply_reader = vtk.vtkPLYReader()
        self.mapper = vtk.vtkPolyDataMapper()
        self.SetMapper(self.mapper)

    @staticmethod
    def vtk_poly_data_from_bytes(byte_str):
        ply_reader = vtk.vtkPLYReader()
        ply_reader.SetInputString(byte_str)
        ply_reader.ReadFromInputStringOn()
        ply_reader.Update()

        return ply_reader.GetOutput()

    @staticmethod
    def load_vtk_poly_data(ply_path):
        if not os.path.exists(ply_path):
            raise FileNotFoundError('File does not exist:', ply_path)

        ply_reader = vtk.vtkPLYReader()
        ply_reader.SetFileName(ply_path)
        ply_reader.Update()
        return ply_reader.GetOutput()

    def set_ply_path(self, ply_path):
        self.ply_reader.SetFileName(ply_path)
        self.ply_reader.Update()

        # self.mapper.SetInputConnection(self.ply_reader.GetOutputPort())
        vtk_poly_data = self.ply_reader.GetOutput()
        self.set_poly_data(vtk_poly_data)

    def set_poly_data(self, vtk_poly_data):
        """Set vtkPolyData directly (for re-using loaded ply)
        """
        self.mapper.SetInputData(vtk_poly_data)
        self.mapper.ScalarVisibilityOff()
        self.mapper.Update()

    def get_poly_data(self):
        return self.mapper.GetInput()

    def get_number_of_points(self):
        return self.get_poly_data().GetNumberOfPoints()

    def set_transform(self, transform):
        self.vtk_transform = vtk.vtkTransform()
        self.vtk_transform.SetMatrix(transform.flatten())
        self.SetUserTransform(self.vtk_transform)

    def add_to_renderer(self, vtk_renderer):
        vtk_renderer.AddActor(self)

    def remove_from_renderer(self, vtk_renderer):
        vtk_renderer.RemoveActor(self)