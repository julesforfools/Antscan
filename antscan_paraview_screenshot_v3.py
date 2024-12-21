import sys
import os

# import paraview
if sys.platform == "win32":
    sys.path.insert(0,"C:\\Program Files\\ParaView 5.11.2\\bin\\Lib")
    sys.path.insert(0,"C:\\Program Files\\ParaView 5.11.2\\bin\\Lib\\site-packages")
else:
    #sys.path.insert(0,'/opt/ParaView-5.11/lib')
    sys.path.insert(0,'/mnt/LSDF/antscan/ParaView-5.12.0-MPI-Linux-Python3.10-x86_64/lib')
    #sys.path.insert(0,'/opt/ParaView-5.11/lib/python3.9/site-packages')
    sys.path.insert(0,'/mnt/LSDF/antscan/ParaView-5.12.0-MPI-Linux-Python3.10-x86_64/lib/python3.10/site-packages')
from paraview.simple import *

if __name__ == "__main__":
    mesh = sys.argv[1]
    camy = int(sys.argv[2])
    #camz = int(sys.argv[3])
    path_to_save = mesh.replace('.stl', '.png')
    mode = 'label'

    Connect()
    #### disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()

    # create a new 'Legacy VTK Reader'
    #Paraview = LegacyVTKReader(registrationName='first', FileNames=[mesh])
    Paraview = STLReader(registrationName='first', FileNames=[mesh])

    # set active source
    SetActiveSource(Paraview)

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')

    # show data in view
    ParaviewDisplay = Show(Paraview, renderView1, 'GeometryRepresentation')

    # get color transfer function/color map for 'scalars'
    scalarsLUT = GetColorTransferFunction(mode)

    # trace defaults for the display properties.

    ParaviewDisplay.Representation = 'Surface'
    ParaviewDisplay.ColorArrayName = ['CELLS', mode]
    ParaviewDisplay.LookupTable = scalarsLUT
    ParaviewDisplay.SelectTCoordArray = 'None'
    ParaviewDisplay.SelectNormalArray = 'None'
    ParaviewDisplay.SelectTangentArray = 'None'
    ParaviewDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    ParaviewDisplay.SelectOrientationVectors = 'None'
    ParaviewDisplay.ScaleFactor = 42.19332799911499
    ParaviewDisplay.SelectScaleArray = mode
    ParaviewDisplay.GlyphType = 'Arrow'
    ParaviewDisplay.GlyphTableIndexArray = mode
    ParaviewDisplay.GaussianRadius = 2.10966639999557497
    ParaviewDisplay.SetScaleArray = [None, '']
    ParaviewDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    ParaviewDisplay.OpacityArray = [None, '']
    ParaviewDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    ParaviewDisplay.DataAxesGrid = 'GridAxesRepresentation'
    ParaviewDisplay.PolarAxes = 'PolarAxesRepresentation'

    # show color bar/color legend
    ParaviewDisplay.SetScalarBarVisibility(renderView1, False)

    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # Hide the scalar bar for this color map if no visible data is colored by it.
    ParaviewDisplay.RescaleTransferFunctionToDataRange(False,True)

    # Hide orientation axes
    renderView1.OrientationAxesVisibility = 0

    # reset view to fit data
    renderView1.ResetCamera()

    # get layout
    layout1 = GetLayout()

    # layout/tab size in pixels
    factor = 10
    yres, xres = factor*425, factor*238
    layout1.SetSize(425, 238)

    # current camera placement for renderView1
    renderView1.CameraPosition = [0, -camy, 0]
    #renderView1.CameraFocalPoint = [0, 0, 0]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 0#310.70784564812436

    # save screenshot
    SaveScreenshot(path_to_save, renderView1, ImageResolution=[yres, xres],
        OverrideColorPalette='WhiteBackground')

    # Reset the session
    Disconnect()
