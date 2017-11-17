from scipy.misc import lena
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from PIL import ImageDraw
W,H = 60,40;
import os
dir = 'D:\\Documents\\CS229\\Project\\SingleLetterDatabase\\'
from PIL import ImageFilter
def randomDistortion(img, key):
    if(key == 0):
        img = np.array(img);
        A = img.shape[0] / 5.0
        w = 2.0 / img.shape[1]

        shift = lambda x: A * np.sin(2.0*np.pi*x * w)

        for i in range(img.shape[1]):
            img[:,i] = np.roll(img[:,i], int(shift(i)))
    elif(key == 1):
        img = img.filter(ImageFilter.BLUR)
    elif(key == 2):
        draw = ImageDraw.Draw(img)
        # 3 generate random origin
        x = np.random.randint(0, W, 1)
        y = np.random.randint(0, H, 1)
        r = np.random.randint(0, 20, 1);
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(2, 100, 2, 127))
    elif(key == 3):
        draw = ImageDraw.Draw(img)
        # 3 generate random origin
        draw.line((0, 0) + img.size, fill=128)
        draw.line((0, img.size[1], img.size[0], 0), fill=128)
    return img;


# for i in os.listdir(dir):
#
#     img =Image.open(dir+i);
#     img = np.array(img)
#     print(img.shape)
#     A = img.shape[0] / 5.0
#     w = 2.0 / img.shape[1]
#
#     shift = lambda x: A * np.sin(2.0*np.pi*x * w)
#
#     for i in range(img.shape[1]):
#         img[:,i] = np.roll(img[:,i], int(shift(i)))
#     plt.figure()
#     plt.imshow(img)
# plt.show()

##blur
for i in os.listdir(dir):

    img = Image.open(dir+i);
    img = img.filter(ImageFilter.BLUR)
    draw = ImageDraw.Draw(img)
    # 3 generate random origin
    draw.line((0, 0) + img.size, fill=128)
    draw.line((0, img.size[1], img.size[0], 0), fill=128)
    img = np.array(img)
    plt.figure()
    plt.imshow(img)
plt.show()