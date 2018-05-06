from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Activation, Flatten, Conv2D, MaxPooling2D
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import keras
import numpy as np
import cv2
import os

def getData():
    print('資料取得中...')
    dataset = []

    for myDir in tqdm(os.listdir('CroppedYale')):
        for myFile in os.listdir(f'CroppedYale/{myDir}'):
            if not myFile.endswith('.pgm'): continue
            label = myDir
            data = cv2.resize(cv2.imread(f'CroppedYale/{myDir}/{myFile}'), (224, 224)).astype(np.float32)
            dataset.append((label, data))

    return dataset

def VGG_16(classify_num, weights_path=None, input_shape=[224, 224, 3]):
    img_input = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)


    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fca')(x)
    x = Dense(4096, activation='relu', name='fcb')(x)
    x = Dense(classify_num, activation='softmax', name='Classification')(x)


    inputs = img_input
    # Create model.
    model = Model(inputs=inputs, outputs=x, name='vgg16')

    if weights_path:
        model.load_weights(weights_path, by_name=True)

    return model

if __name__ == "__main__":
    dataset = list(getData())
    trainingSet = []
    testingSet = []

    for label in os.listdir('CroppedYale'):
        data = list(filter(lambda x: x[0] == label, dataset))
        trainData, testData = data[:35], data[35:]
        trainingSet.extend(trainData)
        testingSet.extend(testData)

    print('輸入訓練資料中...')

    X = np.zeros((len(trainingSet), 224, 224, 3))
    Y = [0] * len(trainingSet)
    for idx, (y, x) in enumerate(tqdm(trainingSet)):
        X[idx] = x
        Y[idx] = int(y[5:])
    
    encoder = OneHotEncoder()
    labels = encoder.fit_transform(list(map(lambda x: [x], Y))).toarray()

    model = VGG_16(classify_num=labels.shape[1], weights_path='model.h5')
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=1e-4), metrics=['acc'])
    model.fit(x=X, y=labels, epochs=1)