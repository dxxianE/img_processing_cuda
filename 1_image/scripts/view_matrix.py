import matplotlib.pyplot as plt
import numpy as np
import sys

imgName = str(sys.argv[1])
filein  = imgName + ".txt"

img = np.loadtxt(filein)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.show()
