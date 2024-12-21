import sys, os
import glob
import subprocess
import tifffile
import math

print(sys.argv[0:])
#print(len(sys.argv))
#sys.argv is a list in Python, which contains the command-line arguments passed to the script.
#sys.argv[0] is the script, sys.argv[1] is the STL file directory,
#sys.argv[2] is the TIFF file directory

if __name__ == "__main__":
    paths = glob.glob(sys.argv[1]+'/*.stl')
    for path_to_mesh in paths:
        # Need to adjust for mesh size
        path_to_tiff = os.path.join(sys.argv[1], os.path.basename(path_to_mesh).replace('.stl','.tif'))
        print(path_to_tiff)
        # Use the image path to feel out the voxel size and adjust using the helper function
        if '/10x/' in path_to_tiff:
            res = 1.22
        elif '/5x/' in path_to_tiff:
            res = 2.44
        elif '/2x/' in path_to_tiff:
            res = 6.11
        elif '/GAGA_10x/' in path_to_tiff:
            res = 1.22
        elif '/GAGA_5x/' in path_to_tiff:
            res = 2.44
        elif '/CSOSZ_5x/' in path_to_tiff:
            res = 2.44
        elif '/GAGA_2x/' in path_to_tiff:
            res = 6.11
        elif '/CT-Lab/' in path_to_tiff:
            res = 8.20
        #Windows tests:
        if '\\5x\\' in path_to_tiff:
            res = 2.44
        elif '\\2x\\' in path_to_tiff:
            res = 6.11
        id = os.path.basename(path_to_mesh).replace('.stl','')
        path_to_png = path_to_mesh.replace('.stl','.png')
        print(id)
        if os.path.exists(path_to_mesh) and not os.path.exists(path_to_png):
            with tifffile.TiffFile(path_to_tiff) as tif:
                nos = len(tif.pages)
            print("Specimen" , id, "consists of" , nos, "slices")
            camy = nos*2
            #camy = '1000' if nos<1000 else '4000' #more is less, zoom is actually camera location
            camy = math.ceil(camy*res)
            #subprocess.Popen([sys.executable, 'D:\\OneDrive\\University_Stuff\\phd\\script\\antscan_screenshots\\antscan_paraview_screenshot_v3.py', path_to_mesh, str(camy)]).wait()
            subprocess.Popen([sys.executable, 'antscan_paraview_screenshot_v3.py', path_to_mesh, str(camy)]).wait()
