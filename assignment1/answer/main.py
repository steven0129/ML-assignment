from skimage import io
from tqdm import tqdm
from config import Env
import numpy as np
import func
import os
import fire

options = Env()

def train(**kwargs):
    dataset = list(func.dataset())
    trainingSet = testingSet = []

    for label in os.listdir('CroppedYale'):
        data = list(filter(lambda x: x[0] == label, dataset))
        (trainData, testData) = (data[:35], data[35:])
        trainingSet.extend(trainData)
        testingSet.extend(testData)

    correct = count = 0
    for testData in tqdm(testingSet):
        (label, data) = testData
        result = func.kNearestNeighbor(
            trainingSet, testData, k=options.k, distance_type=options.distance_type)
        correct = correct + 1 if label == result else correct
        count = count + 1
        tqdm.write('正確分類: {0}/{1}, 準確率: {2}%'.format(correct,
                                                    count, float(correct)/float(count)*100))

if __name__=='__main__':
    fire.Fire()