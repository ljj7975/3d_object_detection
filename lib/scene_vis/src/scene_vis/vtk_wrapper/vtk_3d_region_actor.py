import vtk

class Vtk3DRegionActor(vtk.vtkActor):
    """VtkActor for displaying arbitrary 3D region"""

    def __init__(self, vertices, colour, opacity = None):

        cps = vtk.vtkConvexPointSet()
        points = vtk.vtkPoints()
        for idx, point in enumerate(vertices):
            points.InsertNextPoint(point)

            cps.GetPointIds().InsertId(idx, idx)

        ug = vtk.vtkUnstructuredGrid()
        ug.Allocate(1, 1)
        ug.InsertNextCell(cps.GetCellType(), cps.GetPointIds())
        ug.SetPoints(points)

        self.mapper = vtk.vtkDataSetMapper()
        self.mapper.SetInputData(ug)

        # actor = vtk.vtkActor()
        self.SetMapper(self.mapper)

        self.GetProperty().SetColor(colour)
        if opacity is not None:
            self.GetProperty().SetOpacity(opacity)
        self.GetProperty().SetLineWidth(3)
        self.GetProperty().EdgeVisibilityOn()
