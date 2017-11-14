'''
Created on Nov 13, 2017

@author: yiliu, yijun
'''

import numpy as np
from PIL import Image
import math
import os

def batchLoadImage(dirName, imageSize = 60*40*3):
    # Load all images in a given directory
    # Output: dataset is an m-by-n array,
    # where m is number of examples and n is dimension
    # So the i-th example is dataset[i]

    counter = 0
    dataset = np.array([])
    # Each image should be 60*40, with RGB 3 channels
    for fileName in os.listdir(dirName):
        if fileName.endswith('.png'):
            data = loadImage(os.path.join(dirName, fileName))
            if(np.size(data) != imageSize):
                continue
            dataset = np.append(dataset, data)
            counter += 1
    dataset = dataset.reshape(counter, int(np.size(dataset)/counter))
    return dataset

def loadImage(imgFileName):
    # Load one image with given file name
    # Output: data is an 1-by-n vector, where n is dimension

    img = Image.open(imgFileName)

    # Preprocess the image (e.g. denoise)
    img = preprocessImg(img)

    # data is arranged first in columns. e.g. a 160(length)x60(height) image becomes a (60,160,3) array
    data = np.array(img, dtype = 'uint8')
    # Flattern data into an 1-by-n vector
    data = flatternData(data)
    return data

def toGrayscale(img):
    # To grayscale image, this lowers the dimension of img

    grayscaleImg = img.convert('L')
    return grayscaleImg

def toBW(img):
    # To black-and-white image, this lowers the dimension of img

    bwImg = img.convert('1')
    return bwImg

def preprocessImg(img):
    # TODO: rewrite preprocessing code

    preprocessedImg = toGrayscale(img);
    return preprocessedImg

def flatternData(data):
    # Flattern a high-dimensional data into an 1-by-n vector

    dim = np.size(data)
    flatternedData = data.reshape(dim)
    return flatternedData

def binarizeData(data, threshold = 127):
    # Binarize data array by threshold

    binarizedData = (data > threshold) * 1
    return binarizedData

# Examples
# data = loadImage('./TrainSet/ACKG.png')
# dataset = batchLoadImage('./TrainSet')
