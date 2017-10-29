import pickle
from keras.models import Sequential
from keras.layers import Dense,Dropout, Activation,Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils

dir='D:\\Documents\\CS229\\Project\\ConvolutionalNeuralNets\\'
data = pickle.load(open(dir+'singlechar_database.p', 'rb'));

images = data[0];
print(images.shape)
features = data[1];
print(images[0].shape)
print(features)
#convert all chars to integers
features = np.array([ord(i) for i in features]);
features = features-65
y_train = np_utils.to_categorical(features);
print(y_train);
print(y_train.shape)
#categorize the features

#check image
# images = images.astype('float32');
# images = images/255.0;
X = images;
plt.imshow(X[1,:,:,:])

#check feature categorizations
d = y_train.shape;
for i in range(d[1]):
    plt.plot(y_train[:,i])
plt.show()