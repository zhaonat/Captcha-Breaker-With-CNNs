from scipy.misc import lena
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image
import os
dir = 'D:\\Documents\\CS229\\Project\\SingleLetterDatabase\\'
from PIL import ImageFilter

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
    img = np.array(img)
    plt.figure()
    plt.imshow(img)
plt.show()