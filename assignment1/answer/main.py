from skimage import io
import numpy as np
import func
import os

dataset = list(func.dataset())
trainingSet = []
testingSet = []

for label in os.listdir('CroppedYale'):
    data = list(filter(lambda x: x[0]==label, dataset))
    (trainData, testData) = (data[:35], data[35:])
    trainingSet.extend(trainData)
    testingSet.extend(testData)

result = func.kNearestNeighbor(trainingSet, testingSet[0], k=9)
print(result)