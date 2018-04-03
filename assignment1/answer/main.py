from skimage import io
from tqdm import tqdm
from config import Env
from random import *
import numpy as np
import func
import os
import fire

options = Env()


def trainingProcess(k=1, distance_type='ssd'):
    dataset = list(func.dataset())
    trainingSet = []
    testingSet = []

    for label in os.listdir('CroppedYale'):
        data = list(filter(lambda x: x[0] == label, dataset))
        trainData, testData = data[:35], data[35:]
        trainingSet.extend(trainData)
        testingSet.extend(testData)

    correct = count = 0
    for testData in tqdm(testingSet):
        label, _ = testData
        result = func.kNearestNeighbor(
            trainingSet, testData, k=k,
            distance_type=distance_type)

        correct = correct + 1 if label == result else correct
        count = count + 1
        tqdm.write('正確分類: {0}/{1}, 準確率: {2}%'.format(correct,
                                                     count, float(correct)/float(count)*100))

    return correct, count, float(correct)/float(count)*100


def train():
    correct, count, acc = trainingProcess(k=options.k, distance_type=options.distance_type)
    print('正確分類: {0}/{1}, 準確率: {2}%'.format(correct, count, acc))


def hyperparameter():
    with open('hyperparameter.csv', 'a') as f:
        result = []
        total_distance_type = ['sad', 'ssd']
        for i in tqdm(range(230)):
            randomK = randint(1, 500)
            randomD = total_distance_type[randint(0, 1)]
            tqdm.write('開始Sample第{0}次 (k={1}, randomD={2})'.format(i + 1, randomK, randomD))
            _, _, acc = trainingProcess(k=randomK, distance_type=randomD)
            result.append({'acc': acc, 'k': randomK, 'distance_type': randomD})
            f.write(
                '\n{0}, {1}, {2}'.format(randomK,randomD,acc))

        maxObj = max(result, key=lambda x: x['acc'])
        print('最高準確率: {0} (k={1}, distance_type={2})'.format(
            maxObj['acc'], maxObj['k'], maxObj['distance_type']))

if __name__ == '__main__':
    fire.Fire()
