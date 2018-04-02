import numpy as np
from skimage import io
import os

def dataset():
    dataset = []

    for myDir in os.listdir('CroppedYale'):
        for myFile in os.listdir('CroppedYale/{}'.format(myDir)):
            label = myDir
            img = io.imread('CroppedYale/{0}/{1}'.format(myDir, myFile))
            data = np.array(img)
            dataset.append((label, data))

    return dataset

def kNearestNeighbor(train, test, k=1, distance_type='sad'):
    train = list(map(lambda data: {'label': data[0], 'data': data[1]}, train))
    test = {'label': test[0], 'data': test[1]}
    diff = lambda x, y: x.astype(float) - y.astype(float)

    if distance_type == 'sad':
        distance = lambda x: np.sum(abs(diff(x['data'], test['data'])))
    elif distance_type == 'ssd':
        distance = lambda x: np.sum(diff(x['data'], test['data'])**2)

    topK = list(sorted(train, key=distance))[:k]
    topLabels = [topK[i]['label'] for i in range(len(topK))]
    return max(set(topLabels), key=lambda x: topLabels.count(x))
