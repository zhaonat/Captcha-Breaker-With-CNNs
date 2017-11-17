import pickle
import numpy as np

def load_images_labels(filepath):
    data = pickle.load(open(filepath, 'rb'));

    images = data[0];
    print(images.shape)
    labels = data[1];
    print(images[0].shape)
    # convert all chars to integers
    labels = np.array([ord(i) for i in labels]);
    labels = labels - 65;

    return [images, labels]