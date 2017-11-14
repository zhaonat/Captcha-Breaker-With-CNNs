import pickle
from keras.models import Sequential
from keras.layers import Dense,Dropout, Activation,Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
dir= 'D:\\Documents\\CS229\\Project\\ConvolutionalNeuralNets\\'

data = pickle.load(open(dir+'singlechar_database.p', 'rb'));

images = data[0];
print(images.shape)
features = data[1];
print(images[0].shape)
print(features)
#convert all chars to integers
features = np.array([ord(i) for i in features]);
features = features-65;
y_train = np_utils.to_categorical(features);
print(y_train);
print(y_train.shape)
#categorize the features

## process images
## convert X to flaots
# images = images.astype('float32');
# images = images/255.0;
X = images;
# X = X[:,:,:,0:2];
X = X[:,:,:,0];
X = np.reshape(X, (5000,60,40,1));
## Create the  convolutional neural net


model = Sequential();

## generate sequence of layers
#first convolution layer to process imag
#first argument is filters, the dimensionality of output space
#second argument is kernel size, or size of conv window
#third argumen
model.add(Conv2D(20,(2,2),strides = 1, activation = 'relu', input_shape = (60,40,1)))
model.add(Dropout(0.1));
model.add(Conv2D(40,(5,5),strides = 1, activation = 'relu'))
model.add(Dropout(0.1));
# model.add(MaxPooling2D(pool_size = (2,2))); #pool size = reduction factor
# model.add(Conv2D(80,(6,6),strides = 1, activation = 'relu'))
# model.add(Dropout(0.1));
# model.add(Conv2D(160,(2,2),strides = 1, activation = 'relu'))
# model.add(Dropout(0.1));
# model.add(MaxPooling2D(pool_size = (2,2))); #pool size = reduction factor
# model.add(Conv2D(80,(2,2),strides = 1, activation = 'relu'))
# model.add(Dropout(0.1));
# model.add(Conv2D(40,(3,3),strides = 1, activation = 'relu'))
# # maxpooling essentially does a dimensionality reduction
model.add(MaxPooling2D(pool_size = (2,2))); #pool size = reduction factor

model.add(Flatten()) #flattens the input (so it's 1d after this point)
#Dense is just a normal neural network layer (the core layer)

## only after convolution do we start appending dense layers like a NN



model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.01));
num_classes = y_train.shape[1]
model.add(Dense(num_classes, activation = 'softmax'));


## compile the model means all the layers are compiled into the final network
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

## fit the model to the train
#careful about batch size, can lead to nonetype is not callable error
history = model.fit(X, y_train, validation_split=0.05, batch_size=200, epochs=500, verbose=1);
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()