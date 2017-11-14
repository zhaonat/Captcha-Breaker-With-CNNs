## script which covers letters with opaque or partially opaque shapes

from scipy.misc import lena
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image
import os
dir = 'D:\\Documents\\CS229\\Project\\SingleLetterDatabase\\'
from PIL import ImageFilter
from PIL import ImageDraw

[W,H] = [40,60];
for i in os.listdir(dir):

    img = Image.open(dir+i);
    draw = ImageDraw.Draw(img)
    #3 generate random origin
    x = np.random.randint(0,W,1)
    y = np.random.randint(0,H,1)
    r = np.random.randint(0, 20, 1);
    draw.ellipse((x - r, y - r, x + r, y + r),fill=(2,100,2,127))
    img = np.array(img)
    plt.figure()
    plt.imshow(img)
plt.show()