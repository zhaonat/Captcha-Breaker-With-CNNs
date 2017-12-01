import numpy as np;
import matplotlib.image as img
import os;
import pickle;

dir = 'D:\\Documents\\CS229\\Project\\DATABASES\\CaptchaDatabase\\'
dir = 'D:\\Documents\\CS229\\Project\\DATABASES\\SingleCharProcessed\\'
dir = 'D:\\Documents\\CS229\\Project\\DATABASES\\TestSingleChar\\'
dir2='D:\\Documents\\CS229\\Project\\pickleDataSets\\'
imageData = list();
imageLabels = list();
for file in os.listdir(dir):
    print(file)
    label = file[0];
    image = img.imread(dir+file);
    #convert image to grayscale:


    print(label)
    imageData.append(image);
    imageLabels.append(label);

imageData = np.array(imageData);

#pickle.dump((imageData, imageLabels), open(dir2+"singlechar_database.p", 'wb'));
pickle.dump((imageData, imageLabels), open("singlechar_test_database.p", 'wb'));

