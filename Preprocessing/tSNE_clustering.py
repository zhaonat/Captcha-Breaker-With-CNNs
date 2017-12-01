import numpy as np
from sklearn.manifold import TSNE
import dev_constants as dev
import load_pickle_database as ldp
import matplotlib.pyplot as plt
from keras.utils import np_utils
## import a dataset;
filepath = dev.MY_PROJECT_PATH+'\\ConvolutionalNeuralNets\\singlechar_database.p';
images,labels = ldp.load_images_labels(filepath);
print(images.shape)
num_samples = 500;
## generate a representative sample
samples = np.random.randint(0,50000,num_samples);

X = images[samples,:,:,1];

X = np.reshape(X, (num_samples, 2400));
print(X.shape)
labels = np.array([ord(i) for i in labels]);
labels = labels - 65;
y = labels
y = y[samples];
print('fitting')
X_embedded = TSNE(n_components=2).fit_transform(X)
print(X_embedded.shape)

plt.figure;
plt.scatter(X_embedded[:,0], X_embedded[:,1], c = y);
plt.show();
