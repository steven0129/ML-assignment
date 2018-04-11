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

    correct = 0
    count = 0
    for testData in tqdm(testingSet):
        label, _ = testData
        result = func.kNearestNeighbor(
            trainingSet, testData, k=k,
            distance_type=distance_type)

        # correct = correct + 1 if label == result else correct
        if label==result: correct += 1
        count = count + 1
        tqdm.write('正確分類: {0}/{1}, 準確率: {2}%'.format(correct,
                                                     count, float(correct)/float(count)*100))

    return correct, count, float(correct)/float(count)*100


def train(**kwargs):
    for k_,v_ in kwargs.items():
        setattr(options,k_,v_)

    correct, count, acc = trainingProcess(k=options.k, distance_type=options.distance_type)
    print('正確分類: {0}/{1}, 準確率: {2}%'.format(correct, count, acc))

if __name__ == '__main__':
    fire.Fire()
