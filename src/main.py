import numpy as np
import random
from src.Setup import initDataset, loadImages, Constants
from src.model.Classifiers import SoftMax
from src.model.HyperParams import HyperParams
from src.CNN import Model

STEP_SIZE = 1e-7
FEATURE_STEP_SIZE = 1e-7
REG = 1e-3
BATCH_SIZE = 50


def doTheStuff():
    data = initDataset()

    inputSize = data[0].getData().shape[1]

    # randomize data for better distribution
    random.shuffle(data)
    datasetSize = len(data)

    # initialize data
    datasetValues = np.empty((datasetSize, Constants.CHANNEL_SIZE, inputSize, inputSize), dtype=int)
    datasetLabels = np.empty((datasetSize, 1), dtype=int)
    position = 0
    for value in data:
        datasetValues[position] = value.getData()
        datasetLabels[position] = value.getLabel()
        position += 1

    trainingDataset = datasetValues[0:400], datasetLabels[0:400]
    validatingDataset = datasetValues[400:], datasetLabels[400:]
    hyperParams = HyperParams(STEP_SIZE, REG, FEATURE_STEP_SIZE)

    params = {'receptiveFieldSize': 3, 'stride': 1, 'zeroPadding': None, 'f_number': 5}

    # model getting trained
    model = Model(inputSize * inputSize, SoftMax(BATCH_SIZE, hyperParams), hyperParams, params, BATCH_SIZE)
    model.train(trainingDataset)
    model.validate(validatingDataset)
    return model


def play():
    model = doTheStuff()
    while (True):
        input('Press anything to predict')
        print('predicting... ')


doTheStuff()
