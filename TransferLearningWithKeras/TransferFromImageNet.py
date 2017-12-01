from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.layers import Conv2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import numpy as np
from sklearn.model_selection import train_test_split
import dev_constants as dev
from keras.utils import np_utils

img_width, img_height = 256, 256
train_data_dir = "data/train"
validation_data_dir = "data/val"
nb_train_samples = 4125
nb_validation_samples = 466


model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
#print(model.summary())

## step 1: freeze layers that we do not want to tune
for layer in model.layers[0:5]:
    layer.trainable = False


## add custom layers (which we tune so we can train on our captchas)
## one of the layers that we must change is the input layer!
input = Input(shape =  (60,40,3), name = 'captcha_input')
output_vgg16_conv = model(input);
## add in some extra layers into the output

x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(26, activation="softmax")(x)

# creating the final model
model_final = Model(input = input, output = predictions)

# compile the model
model_final.compile(loss = "categorical_crossentropy",\
                    optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
print(model_final.summary())
## test the model on our captchas data-set
import load_pickle_database as pdb
dir= dev.MY_PROJECT_PATH+'\\ConvolutionalNeuralNets\\'

path = dir+'singlechar_database.p'

images, features = pdb.load_images_labels(path);
X = images;
# X = X[:,:,:,0:2];
X = np.reshape(X, (50000,60,40,3));
labels = np.array([ord(i) for i in features]);
labels = labels - 65;
y = np_utils.to_categorical(labels);

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2);

history = model_final.fit(X_train, y_train, validation_split=0.2, batch_size=400, epochs=70, verbose=1);
print(model_final.evaluate(X_test, y_test))
