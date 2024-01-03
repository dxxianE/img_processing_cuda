from matplotlib.image import imread
import numpy as np
import sys

imgName = str(sys.argv[1])
filein  = imgName + ".jpg"
fileout = imgName + ".txt"

img = imread(filein)
gray = img[:,:,0]
print("Size image:",len(gray[0,:]),len(gray[:,0]))

np.savetxt(fileout, gray, fmt="%d")
