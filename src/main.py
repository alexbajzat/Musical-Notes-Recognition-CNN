import numpy as np
import random
from src.data.Setup import initDataset, Constants
from src.data.mnistdata import initMNISTDataset
from src.model.Classifiers import SoftMax
from src.model.HyperParams import HyperParams
from src.CNN import Model

STEP_SIZE = 1e-5
FEATURE_STEP_SIZE = 1e-5
REG = 1e-3
BATCH_SIZE = 100


def doTheStuff(data):

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

    params = {'receptiveFieldSize': 3, 'stride': 1, 'zeroPadding': None, 'f_number': 10}

    # model getting trained
    model = Model(inputSize * inputSize, SoftMax(hyperParams), hyperParams, params, BATCH_SIZE)
    model.train(trainingDataset)
    model.validate(validatingDataset)
    return model


def trainWithMnist():
    data = initMNISTDataset()
    doTheStuff(data)

def train():
    data = initDataset()

    doTheStuff(data)

def play():
    model = doTheStuff()
    while (True):
        input('Press anything to predict')
        print('predicting... ')

train()
