from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.layers import Conv2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import pickle

img_width, img_height = 256, 256
train_data_dir = "data/train"
validation_data_dir = "data/val"
nb_train_samples = 4125
nb_validation_samples = 466
batch_size = 16
epochs = 50

model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
print(model.summary())

## step 1: freeze layers that we do not want to tune
for layer in model.layers[1:5]:
    layer.trainable = False


## add custom layers (which we tune so we can train on our captchas)
## one of the layers that we must change is the input layer!
y = model.input; ## STILL NEEDS WORK
y.add(Conv2D(20,(2,2),strides = 1, activation = 'relu', input_shape = (60,40,1)))

## add in some extra layers into the output
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(26, activation="softmax")(x)

# creating the final model
model_final = Model(input = y, output = predictions)

# compile the model
model_final.compile(loss = "categorical_crossentropy",\
                    optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

## test the model on our captchas data-set
