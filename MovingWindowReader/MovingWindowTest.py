import dev_constants as dev;
import os;
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

dir = dev.MY_PROJECT_PATH+'\\CaptchaDatabase\\'
neuraldir = dev.MY_PROJECT_PATH+'\\NeuralNetTesting\\'
def MovingWindow(img, steplength=10):
    W = 40;
    d = img.shape;
    columns = d[1];
    start = 0;
    imgStore = list();
    for i in range(0,columns, steplength):

        window = img[:,start+i:W+i];

        imgStore.append(window);

    return imgStore;

## load a neural net
model = load_model(neuraldir+'single_char_CNN.h5')
stopcount = 0;
for imgfile in os.listdir(dir):
    img = Image.open(dir+imgfile);
    label = imgfile.strip('.png');
    print(img.size)
    imgStore = MovingWindow(np.array(img), steplength=20);

    captchaBatch = list();
    for i in imgStore:
        #print(i.shape)
        if(i.size!= 60*40*3):
            continue;
        captchaBatch.append(np.reshape(i[:,:,1], (60,40,1)));

        ## look at the image
        # plt.imshow(i);
        # #plt.show()

    captchaBatch = np.array(captchaBatch);
    print(captchaBatch.shape)
    prediction = model.predict_classes(captchaBatch, batch_size = 13, verbose = 1);
    print(prediction)
    proba= model.predict_proba(captchaBatch, batch_size = 13, verbose = 1);
    print(proba)
    #extract max probability from each row
    # for j in range(proba.shape[0]):
    #     print(np.max(j))
    word = ''
    for i in prediction:
        word = word+(chr(int(i)+65))
    print(word);
    stopcount +=1;

    plt.imshow(img)
    plt.show()
    if(stopcount > 10):
        break;
