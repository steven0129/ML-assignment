from skimage import io
import numpy as np
import func
import os

dataset = list(func.dataset())

for label in os.listdir('CroppedYale'):
    data = list(filter(lambda x: x[0]==label, dataset))
    (trainData, testData) = (data[:35], data[35:])
    func.kNearestNeighbor(trainData, testData[0], k=2)
    
