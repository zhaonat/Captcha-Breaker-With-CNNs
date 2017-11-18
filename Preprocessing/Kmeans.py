## kmeans clustering of images
from sklearn.cluster import KMeans
import load_pickle_database as ldp
import numpy as np
import dev_constants as dev
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split;
kmeans = KMeans(n_clusters=26, random_state=0, verbose=1);

## import a dataset;
filepath = dev.MY_PROJECT_PATH+'\\ConvolutionalNeuralNets\\singlechar_database.p';
images,labels = ldp.load_images_labels(filepath);
print(images.shape)
num_samples = 3000;
## generate a representative sample
samples = np.random.randint(0,50000,num_samples);

X = images[samples,:,:,1];


X = np.reshape(X, (num_samples, 2400));
print(X.shape)

kmeans.fit(X);
print(kmeans.cluster_centers_)

#check what the cluster centers look like
d = kmeans.cluster_centers_.shape;
for i in range(d[0]):
    center_image = kmeans.cluster_centers_[i,:];
    img = np.reshape(center_image, (60,40));
    plt.imshow(img);
    plt.show()