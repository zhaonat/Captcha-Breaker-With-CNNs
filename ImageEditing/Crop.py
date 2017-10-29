# Import Pillow:
from PIL import Image
import matplotlib.pyplot as plt

import os

dir = 'D:\\Documents\\CS229\\Project\\SingleCharDatabase\\'
outdir = 'D:\\Documents\\CS229\\Project\\SingleCharProcessed\\'
counter = 0;
for image in os.listdir(dir):
    img = Image.open(dir+image);
    print(type(img));
    img2 = img.crop((0, 0, 40,60))
    # plt.figure()
    # plt.imshow(img2)
    img2.save(outdir+image)
    counter+=1;
    print(counter)
    # if(counter > 10):
    #     break;

plt.show()