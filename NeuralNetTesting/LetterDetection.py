## import a saved model and then test
from keras.models import load_model
import dev_constants as dev
import load_pickle_database as ldp;
from keras.utils import np_utils;
from sklearn.model_selection import train_test_split
import numpy as np

## load a prediction/generalization set
file = dev.MY_PROJECT_PATH+'\\SingleCharOverlay\\overlaychar_database.p'
images, features = ldp.load_images_labels(file);
y = features;
X = images;

X = X[:,:,:,0];
X = np.reshape(X, (50000,60,40,1));


y = y[:,1]; #switch this between 0 an d1
labels = np.array([ord(i) for i in y]);
labels = labels - 65;
y = np_utils.to_categorical(labels);

## load model
model = dev.MY_PROJECT_PATH+'\\NeuralNetTesting\\single_char_CNN.h5';
single_model = load_model(model);

print(single_model.evaluate(X,y));