import cv2
import sys
import os
import glob
from skimage import io

basename = os.path.abspath(sys.argv[1])
dirs = [d for d in os.listdir(basename) if "_lowres" not in d]
for dir in dirs:
    files = glob.glob(basename + f'/{dir}/*.png')
    print(dir, "with", len(files), "pngs")
    if len(files)>0:
        if not os.path.exists(basename + f'/{dir}_lowres'):
            os.mkdir(basename + f'/{dir}_lowres')
            for file in files:
                a = io.imread(file)
                ysh, xsh = a.shape
                y_shape = int(round(ysh/xsh*400))
                b = cv2.resize(a, (400, y_shape), interpolation=cv2.INTER_AREA)
                newfile = file.replace(f'/{dir}/', f'/{dir}_lowres/')
                if not os.path.exists(newfile):
                    io.imsave(newfile, b)
