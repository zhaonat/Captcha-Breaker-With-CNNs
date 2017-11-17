# Import Pillow:
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
from PIL import Image

dir = 'D:\\Documents\\CS229\\Project\\pickleDataSets\\'
outdir = 'D:\\Documents\\CS229\\Project\\SingleCharOverlay\\'
counter = 0;
data = pickle.load(open(dir+'singlechar_database.p', 'rb'));
images = data[0];
print(images.shape)
features = data[1];

numSamples = len(features);
newSamples = 50000;
imageData = list();
imageLabels = list();
for i in range(newSamples):
    if(i%500 == 0):
        print('sample: '+str(i));
    ##generate two random indices to mix the letters
    p1 = np.random.randint(0,numSamples);
    p2 = np.random.randint(0,numSamples);

    pic1 = 255*images[p1,:,:,:];
    pic2 = 255*images[p2,:,:,:];
    label1 = features[p1];
    label2 = features[p2];
    combinedLabel = [label1,label2];

    img1 = Image.fromarray(pic1.astype('uint8'), 'RGB')
    img2 = Image.fromarray(pic2.astype('uint8'), 'RGB')
    img = Image.blend(img1, img2, alpha = 0.5);
    #plt.imshow(img)
    imageData.append(np.array(img));
    imageLabels.append(combinedLabel);

imageData = np.array(imageData);
imageLabels = np.array(imageLabels);
print(imageData.shape)
print(imageLabels.shape)
pickle.dump((imageData, imageLabels), open(outdir+"overlaychar_database.p", 'wb'));

