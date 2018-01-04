import numpy as np
import random
from src.Setup import initDataset, loadImages, Constants
from src.model.Classifiers import SoftMax
from src.model.HyperParams import HyperParams
from src.CNN import Model

STEP_SIZE = 1e-07
REG = 1e-3


def doTheStuff():
    data = initDataset()
    datasetSize = len(data)
    inputSize = data[0].getData().shape[1]

    # randomize data for better distribution
    random.shuffle(data)

    # initialize data
    datasetValues = np.empty((datasetSize, Constants.CHANNEL_SIZE, inputSize, inputSize), dtype=int)
    datasetLabels = np.empty((datasetSize, 1), dtype=int)
    position = 0
    for value in data:
        datasetValues[position] = value.getData()
        datasetLabels[position] = value.getLabel()
        position += 1

    dataset = datasetValues, datasetLabels
    hyperParams = HyperParams(STEP_SIZE, REG)

    params = {'receptiveFieldSize': 3, 'stride': 1, 'zeroPadding': None, 'f_number': 5}

    # model getting trained
    model = Model(inputSize * inputSize, SoftMax(datasetSize, hyperParams), hyperParams, params)
    model.train(dataset)
    model.validate(dataset)


doTheStuff()
