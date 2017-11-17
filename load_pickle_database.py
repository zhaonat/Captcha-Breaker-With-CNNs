import pickle
import numpy as np

def load_images_labels(filepath):
    data = pickle.load(open(filepath, 'rb'));

    images = data[0];
    print(images.shape)
    labels = data[1];
    print(images[0].shape)


    return [images, labels]