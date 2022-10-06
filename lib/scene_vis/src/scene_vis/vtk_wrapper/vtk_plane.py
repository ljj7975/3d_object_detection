"""https://github.com/kujason/scene_vis"""

import numpy as np
import vtk


class VtkPlane(vtk.vtkActor):
    """Axis aligned plane"""

    def __init__(self):
        super().__init__()
        self.vtk_plane_source = vtk.vtkPlaneSource()
        self.vtk_plane_mapper = vtk.vtkPolyDataMapper()

    def set_plane(self, axes, offset, extents: np.ndarray):
        """Calculates 3 points for plane visualization
        based on the provided ground plane

        Args:
            axes: "xy", "xz", "yz"
            offset: Offset distance
            extents: Extents along the plane for visualization (2, 2)
        """

        extents = np.asarray(extents)
        assert extents.shape == (2, 2)

        min_i = extents[0][0]
        max_i = extents[0][1]
        min_j = extents[1][0]
        max_j = extents[1][1]

        if axes == "xy":
            plane_points = np.asarray(
                [[min_i, min_j, offset], [max_i, min_j, offset], [min_i, max_j, offset]]
            )
        elif axes == "xz":
            plane_points = np.asarray(
                [[min_i, offset, min_j], [max_i, offset, min_j], [min_i, offset, max_j]]
            )
        elif axes == "yz":
            plane_points = np.asarray(
                [[offset, min_i, min_j], [offset, max_i, min_j], [offset, min_i, max_j]]
            )
        else:
            raise ValueError("Invalid axes", axes)

        self.vtk_plane_source.SetOrigin(*plane_points[0])
        self.vtk_plane_source.SetPoint1(*plane_points[1])
        self.vtk_plane_source.SetPoint2(*plane_points[2])

        self.vtk_plane_source.Update()

        vtk_plane_poly_data = self.vtk_plane_source.GetOutput()
        self.vtk_plane_mapper.SetInputData(vtk_plane_poly_data)

        self.SetMapper(self.vtk_plane_mapper)

    def set_transform(self, transform):
        self.vtk_transform = vtk.vtkTransform()
        self.vtk_transform.SetMatrix(transform.flatten())
        self.SetUserTransform(self.vtk_transform)
