## noise modules from PIL

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
#select a random blob size
def salt_pepper(image):
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1
      return out;

for i in os.listdir(dir):

    img = Image.open(dir+i);
    #img = img.filter(ImageFilter.BLUR)
    imgarr = np.array(img)
    img = salt_pepper(imgarr)
    print(img)
    #im2 = img.filter(ImageFilter.GaussianBlur(1))
    # draw = ImageDraw.Draw(img)
    # # 3 generate random origin
    # draw.line((0, 0) + img.size, fill=128)
    # draw.line((0, img.size[1], img.size[0], 0), fill=128)
    # img = np.array(img)
    # plt.figure()
    plt.imshow(img)
plt.show()