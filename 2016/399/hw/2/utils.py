import os
import numpy as np
import pylab as plt
import _pickle as pickle
from scipy.misc import imread
from matplotlib.colors import ListedColormap

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

def load_cifar10(base='./data/cifar10'):
    def load_CIFAR_batch(filename):
        with open(filename, 'rb') as f:
            datadict = pickle.load(f, encoding='bytes')
            Y = np.array(datadict[b'labels'])
            X = datadict[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            return X, Y

    def load_CIFAR10(ROOT):
        xs, ys = [], []
        for b in range(1, 6):
            f = os.path.join(ROOT, 'data_batch_%d' % (b,))
            X, Y = load_CIFAR_batch(f)
            xs.append(X)
            ys.append(Y)
        Xtr, Ytr = np.concatenate(xs), np.concatenate(ys)
        del X, Y
        Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
        return Xtr, Ytr, Xte, Yte

    # Load the raw CIFAR-10 data
    cifar10_dir = os.path.join(base, 'cifar-10-batches-py')
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    X_val, y_val = X_test, y_test

    return (X_train, y_train, X_val, y_val, X_test, y_test), X_train.shape[0], X_test.shape[0], (None, 3, 32, 32)

def plot_surface(X, y, clf):
    h = 0.2
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 8))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
