from matplotlib.image import imread
import numpy as np
import sys

imgName = str(sys.argv[1])
filein  = imgName + ".jpg"
fileout = imgName + ".txt"

img = imread(filein)
r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
print("Size image:",len(r[0,:]),len(r[:,0]))
gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
np.savetxt(fileout, gray, fmt="%d")
