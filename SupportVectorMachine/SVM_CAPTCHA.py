'''
Created on Nov 13, 2017

@author: Yi & Yijun
'''

import numpy as np
from PIL import Image
import math
import load_data

def loadImage(imgFileName):
    img = Image.open(imgFileName)

    # Preprocess the image (e.g. denoise)
    img = preprocessImg(img)

    # data is flatterned first in columns. e.g. a 160(length)x60(height) image is turned into a 60x160x3 array
    data = np.array(img, dtype = 'int32')
    return data

def preprocessImg(img):
    # TODO: add preprocessing code
    return img

def flatternData(data):
    # Flattern a high-dimensional data into an 1-by-n vector
    dim = np.size(data)
    flatternedData = data.reshape(1, dim)
    return flatternedData

def binarizeData(data, threshold = 127):
    if(data > threshold):
        pass
    return data

#tImg = loadImage('./TrainSet/ACKG.png')
#[a,b,c] = np.shape(getImg)
#print(a,b,c)
#print(getImg.max(), getImg.min())

def processImg(imgFilename):
    imgLoad = loadImage(imgFilename)
    imgPreprocess = preprocessImg(imgLoad)
    imgFlattened = flatternData(imgPreprocess)
    img = binarizeData(imgFlattened)
    return img
    
