import numpy as np
from sklearn.utils import shuffle

import skimage.exposure

import pickle
from pandas.io.parsers import read_csv


def load_pickled_data(file, columns):
    with open(file, mode='rb') as f:
        dataset = pickle.load(f)
    return tuple(map(lambda c: dataset[c], columns))



def preprocess_dataset(X, y=None):
    #convert to grayscale
    X = 0.299 * X[:, :, :, 0] + 0.587 * X[:, :, :, 1] + 0.114 * X[:, :, :, 2]
    #Scale features to be in [0, 1]
    X = (X / 255.).astype(np.float32)

    for i in range(X.shape[0]):
        X[i] = skimage.exposure.equalize_adapthist(X[i])
    
    if (y.any() != None):
        y=np.eye(43)[y]
        # y=to_categorical(y)
        X,y=shuffle(X,y)

    X=X.reshape(X.shape+(1,))
    return X,y

def load_and_process_data(file):

    X, y = load_pickled_data(file, columns = ['features', 'labels'])
    X, y = preprocess_dataset(X, y)
    return (X, y)

def class_name(one_hot):
    signnames = read_csv("traffic_sign_classification\\signnames.csv").values[:, 1]
    return signnames[one_hot.nonzero()[0][0]]
