## kmeans clustering of images
from sklearn.cluster import KMeans
import load_pickle_database as ldp
import numpy as np
import dev_constants as dev
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split;
from sklearn.decomposition import PCA

kmeans = KMeans(n_clusters=26, random_state=0, verbose=1);

## import a dataset;
filepath = dev.MY_PROJECT_PATH+'\\ConvolutionalNeuralNets\\singlechar_database.p';
images,labels = ldp.load_images_labels(filepath);
print(images.shape)
num_samples = 10000;
## generate a representative sample
samples = np.random.randint(0,50000,num_samples);
labels = np.array([ord(i) for i in labels]);
labels = labels - 65;
labels = labels[samples];
X = images[samples,:,:,1];
print(X.shape)
X = np.reshape(X, (num_samples, 2400));
pca = PCA(n_components=2)
X_red = pca.fit_transform(X);
print(X_red.shape)
plt.figure()
plt.scatter(X_red[:,0], X_red[:,1], c = labels );
plt.show();

# kmeans.fit(X);
# print(kmeans.cluster_centers_)
#
# #check what the cluster centers look like
# d = kmeans.cluster_centers_.shape;
# print(d)
# for i in range(d[0]):
#     center_image = kmeans.cluster_centers_[i,:];
#     img = np.reshape(center_image, (60,40));
#     plt.imshow(img);
#     plt.show()