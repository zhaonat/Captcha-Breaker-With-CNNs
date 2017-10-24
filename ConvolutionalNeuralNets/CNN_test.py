import pickle
from keras.models import Sequential
from keras.layers import Dense,Dropout, Activation,Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils

data = pickle.load(open('singlechar_database.p', 'rb'));

images = data[0];
print(images.shape)
features = data[1];
print(images[0].shape)
print(features)
#convert all chars to integers
features = [ord(i) for i in features];
y_train = np_utils.to_categorical(features);
print(y_train);
print(y_train.shape)
#categorize the features

## process images
## convert X to flaots
images = images.astype('float32');
images = images/255.0;
X = images;
## Create the  convolutional neural net


model = Sequential();

## generate sequence of layers
#first convolution layer to process imag
#first argument is filters, the dimensionality of output space
#second argument is kernel size, or size of conv window
#third argumen
model.add(Conv2D(20,(2,2),strides = 1, activation = 'relu', input_shape = (60,160,3)))
model.add(Dropout(0.1));
model.add(Conv2D(20,(2,2),strides = 2, activation = 'relu'))
model.add(Dropout(0.1));
model.add(Conv2D(40,(3,3),strides = 1, activation = 'tanh'))
model.add(Dropout(0.1));
# maxpooling essentially does a dimensionality reduction
model.add(MaxPooling2D(pool_size = (2,2))); #pool size = reduction factor

model.add(Flatten()) #flattens the input (so it's 1d after this point)
#Dense is just a normal neural network layer (the core layer)

## only after convolution do we start appending dense layers like a NN

units = 2024; # dimensionality of the output space, determined by

model.add(Dense(units,activation='relu'))
model.add(Dropout(0.01));
model.add(Dense(512, activation = 'tanh'))
num_classes = y_train.shape[1]
model.add(Dense(num_classes, activation = 'softmax'));


## compile the model means all the layers are compiled into the final network
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

## fit the model to the train
model.fit(X, y_train, batch_size=10, nb_epoch=25, verbose=1);
