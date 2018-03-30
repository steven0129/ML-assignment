import numpy as np
from skimage import io
import os


def SAD(img1, img2):
    x = np.array(img1)
    y = np.array(img2)

    distance = lambda a, b: a.astype(float)-b.astype(float)
    return np.sum(abs(distance(x, y)))


def SSD(img1, img2):
    x = np.array(img1)
    y = np.array(img2)

    distance = lambda a, b: a.astype(float)-b.astype(float)
    return np.sum(abs(distance(x, y))**2)


def dataset():
    dataset = []

    for myDir in os.listdir('CroppedYale'):
        for myFile in os.listdir('CroppedYale/{}'.format(myDir)):
            label = myDir
            img = io.imread('CroppedYale/{0}/{1}'.format(myDir, myFile))
            data = np.array(img)
            dataset.append((label, data))

    return dataset
