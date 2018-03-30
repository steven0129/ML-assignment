import numpy as np
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
            dataset.append((myDir, myFile))

    return dataset
