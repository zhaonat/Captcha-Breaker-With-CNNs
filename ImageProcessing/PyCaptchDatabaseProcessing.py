import numpy as np;
import matplotlib.image as img
import os;
import pickle;

dir = 'D:\\Documents\\CS229\\Project\\CaptchaDatabase\\'
dir = 'D:\\Documents\\CS229\\Project\\SingleCharDatabase\\'
dir2='D:\\Documents\\CS229\\Project\\ImageProcessing\\'
imageData = list();
imageLabels = list();
for file in os.listdir(dir):
    print(file)
    label = file[0];
    image = img.imread(dir+file);
    print(label)
    imageData.append(image);
    imageLabels.append(label);

imageData = np.array(imageData);

pickle.dump((imageData, imageLabels), open(dir2+"singlechar_database.p", 'wb'));

