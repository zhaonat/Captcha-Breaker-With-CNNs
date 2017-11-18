import dev_constants as dev;
import os;
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

dir = dev.MY_PROJECT_PATH+'\\CaptchaDatabase\\'

def MovingWindow(img):
    W = 40;
    d = img.shape;
    columns = d[1];
    start = 0;
    imgStore = list();
    for i in range(0,columns, 10):
        window = img[:,start+i:W+i];
        imgStore.append(window);

    return imgStore;

## load a neural net


for imgfile in os.listdir(dir):
    img = Image.open(dir+imgfile);
    print(img.size)
    imgStore = MovingWindow(np.array(img));
    for i in imgStore:
        plt.imshow(i);
        plt.show()
